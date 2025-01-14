#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <filesystem>
#include <algorithm>
#include "poisson_fft.cuh"

using half = __half;
using namespace nvcuda;

namespace {
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr float RHO_MIN = 1e-4f;
constexpr float PI = 3.14159265358979323846f;
}

#ifndef CHECK_LAST
#define CHECK_LAST() do { auto err=cudaPeekAtLastError(); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)
#endif

__global__ void half_to_float(const half* src, float* dst, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __half2float(src[i]);
}

__global__ void float_to_half(const float* src, half* dst, int N){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) dst[i] = __float2half_rn(src[i]);
}

__global__ void reduce_sum(const float* data, double* out, int N){
    __shared__ double sh[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double v = (i < N) ? static_cast<double>(data[i]) : 0.0;
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}

__global__ void update_rho_wmma(const half* rho_in,
                                half* rho_out,
                                const half* A,
                                const float* phi,
                                int NX, int NY,
                                float mix, float phi_scale)
{
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = tile_x * WMMA_N + tx;
    int y = tile_y * WMMA_M + ty;

    __shared__ float sh_C[WMMA_M * WMMA_N];

    int lane = ty * blockDim.x + tx;
    if (lane < 32){
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c;

        wmma::load_matrix_sync(a, A, WMMA_K);
        const half* tile_ptr = rho_in + (tile_y * WMMA_M * NX + tile_x * WMMA_N);
        wmma::load_matrix_sync(b, tile_ptr, NX);
        wmma::fill_fragment(c, 0.0f);
        wmma::mma_sync(c, a, b, c);
        wmma::store_matrix_sync(sh_C, c, WMMA_N, wmma::mem_row_major);
    }
    __syncthreads();

    if (x < NX && y < NY){
        float rho_old = __half2float(rho_in[y * NX + x]);
        float rho_smooth = sh_C[ty * WMMA_N + tx];
        float phi_val = phi[y * NX + x];
        float updated = rho_old + mix * (rho_smooth - rho_old) + phi_scale * phi_val;
        if (!isfinite(updated)) updated = rho_old;
        updated = fmaxf(updated, RHO_MIN);
        rho_out[y * NX + x] = __float2half_rn(updated);
    }
}

static void init_rho(std::vector<float>& rho, int NX, int NY){
    rho.resize(static_cast<size_t>(NX) * NY);
    float sigma = 0.07f;
    float s2 = 2.0f * sigma * sigma;
    for (int y = 0; y < NY; ++y){
        for (int x = 0; x < NX; ++x){
            float fx = (x + 0.5f) / float(NX);
            float fy = (y + 0.5f) / float(NY);
            float dx1 = fx - 0.30f;
            float dy1 = fy - 0.35f;
            float dx2 = fx - 0.72f;
            float dy2 = fy - 0.60f;

            float rho0 = 1.0f;
            rho0 += 0.7f * expf(-(dx1 * dx1 + dy1 * dy1) / s2);
            rho0 += 0.4f * expf(-(dx2 * dx2 + dy2 * dy2) / s2);
            rho0 += 0.08f * sinf(2.0f * PI * fx) * cosf(2.0f * PI * fy);
            rho0 = fmaxf(rho0, RHO_MIN);
            rho[y * NX + x] = rho0;
        }
    }
}

static void dump_field(const std::vector<float>& data, int NX, int NY,
                       const char* tag, int step)
{
    (void)NX; (void)NY;
    char name[128];
    std::snprintf(name, sizeof(name), "frames/%s_%04d.bin", tag, step);
    FILE* fp = std::fopen(name, "wb");
    if (!fp){
        std::fprintf(stderr, "Failed to write %s\n", name);
        return;
    }
    std::fwrite(data.data(), sizeof(float), data.size(), fp);
    std::fclose(fp);
}

int main(int argc, char** argv){
    int NX = (argc > 1) ? std::atoi(argv[1]) : 128;
    int NY = (argc > 2) ? std::atoi(argv[2]) : 128;
    int STEPS = (argc > 3) ? std::atoi(argv[3]) : 60;
    float mix = (argc > 4) ? std::atof(argv[4]) : 0.25f;
    float phi_scale = (argc > 5) ? std::atof(argv[5]) : 0.05f;

    if (NX % WMMA_N != 0 || NY % WMMA_M != 0){
        std::fprintf(stderr, "NX and NY must be multiples of %d\n", WMMA_N);
        return 1;
    }

    size_t cells = static_cast<size_t>(NX) * NY;
    size_t bytes_f = cells * sizeof(float);
    size_t bytes_h = cells * sizeof(half);

    std::vector<float> h_rho;
    std::vector<float> h_phi(cells);
    init_rho(h_rho, NX, NY);

    half *d_rho=nullptr, *d_rho_next=nullptr, *d_A=nullptr;
    float *d_rho_f=nullptr, *d_phi=nullptr;
    double *d_sum=nullptr;
    CHECK_CUDA(cudaMalloc(&d_rho, bytes_h));
    CHECK_CUDA(cudaMalloc(&d_rho_next, bytes_h));
    CHECK_CUDA(cudaMalloc(&d_rho_f, bytes_f));
    CHECK_CUDA(cudaMalloc(&d_phi, bytes_f));
    CHECK_CUDA(cudaMalloc(&d_sum, sizeof(double)));

    // Copy initial rho to device and convert to half.
    CHECK_CUDA(cudaMemcpy(d_rho_f, h_rho.data(), bytes_f, cudaMemcpyHostToDevice));
    int tb = 256;
    int gb = (int)((cells + tb - 1) / tb);
    float_to_half<<<gb, tb>>>(d_rho_f, d_rho, (int)cells);
    CHECK_LAST();
    CHECK_CUDA(cudaDeviceSynchronize());

    // Build the WMMA smoothing matrix A (row-local blur).
    std::vector<half> h_A(WMMA_M * WMMA_K);
    float alpha = 0.12f;
    for (int i = 0; i < WMMA_M; ++i){
        for (int j = 0; j < WMMA_K; ++j){
            float v = 0.0f;
            if (i == j) v = 1.0f - 2.0f * alpha;
            if (j == i - 1 || j == i + 1) v = alpha;
            if (i == 0 && j == 0) v = 1.0f - alpha;
            if (i == WMMA_M - 1 && j == WMMA_M - 1) v = 1.0f - alpha;
            h_A[i * WMMA_K + j] = __float2half_rn(v);
        }
    }
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), cudaMemcpyHostToDevice));

    Poisson2D poisson;
    poisson.init(NX, NY);

    std::filesystem::create_directories("frames");

    dim3 block(WMMA_N, WMMA_M);
    dim3 grid(NX / WMMA_N, NY / WMMA_M);

    for (int s = 0; s < STEPS; ++s){
        // Convert rho to float for Poisson and output.
        half_to_float<<<gb, tb>>>(d_rho, d_rho_f, (int)cells);
        CHECK_LAST();

        // Mean rho for Poisson solve (zero-mean RHS).
        CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(double)));
        reduce_sum<<<gb, tb>>>(d_rho_f, d_sum, (int)cells);
        CHECK_LAST();
        CHECK_CUDA(cudaDeviceSynchronize());
        double h_sum = 0.0;
        CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
        float mean_rho = static_cast<float>(h_sum / double(cells));

        float dx = 1.0f / float(NX);
        float dy = 1.0f / float(NY);
        poisson.solve(d_rho_f, d_phi, mean_rho, dx, dy);

        // Dump frames (rho and phi) every step.
        CHECK_CUDA(cudaMemcpy(h_rho.data(), d_rho_f, bytes_f, cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_phi.data(), d_phi, bytes_f, cudaMemcpyDeviceToHost));
        dump_field(h_rho, NX, NY, "rho", s);
        dump_field(h_phi, NX, NY, "phi", s);

        update_rho_wmma<<<grid, block>>>(d_rho, d_rho_next, d_A, d_phi, NX, NY, mix, phi_scale);
        CHECK_LAST();
        CHECK_CUDA(cudaDeviceSynchronize());
        std::swap(d_rho, d_rho_next);

        if ((s % 10) == 0 || s == STEPS - 1){
            float rmin = *std::min_element(h_rho.begin(), h_rho.end());
            float rmax = *std::max_element(h_rho.begin(), h_rho.end());
            std::printf("Step %4d/%4d  rho[min,max]=[%.4f, %.4f]\n", s, STEPS, rmin, rmax);
        }
    }

    poisson.destroy();
    cudaFree(d_A);
    cudaFree(d_rho);
    cudaFree(d_rho_next);
    cudaFree(d_rho_f);
    cudaFree(d_phi);
    cudaFree(d_sum);
    return 0;
}
