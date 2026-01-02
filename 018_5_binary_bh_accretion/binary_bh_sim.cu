#include <cuda_runtime.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <curand_kernel.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct SimParams {
    float gm1;
    float gm2;
    float rs1;
    float rs2;
    float separation;
    float omega_orbit;
    float omega_disk;
    float rho0;
    float r0;
    float sigma;
    float arm_amp;
    float arm_m;
    float bin_amp;
    float refine_radius;
    float refine_amp;
    float filter_cutoff;
};

static void check_cuda(cudaError_t err, const char *ctx) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", ctx, cudaGetErrorString(err));
        std::exit(1);
    }
}

static void check_cufft(cufftResult res, const char *ctx) {
    if (res != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cuFFT error at %s: %d\n", ctx, static_cast<int>(res));
        std::exit(1);
    }
}

static void check_cublas(cublasStatus_t status, const char *ctx) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s: %d\n", ctx, static_cast<int>(status));
        std::exit(1);
    }
}

__device__ __forceinline__ float safe_radius(float r, float rs) {
    return fmaxf(r, rs + 1e-3f);
}

__global__ void setup_curand(curandStatePhilox4_32_10_t *states, unsigned long seed, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void noise_kernel(curandStatePhilox4_32_10_t *states, float *noise, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float u = curand_uniform(&states[idx]);
        noise[idx] = 2.0f * u - 1.0f;
    }
}

__global__ void refine_patch(
    float *rho,
    float *phi,
    int nx,
    int ny,
    int start_x,
    int start_y,
    float t,
    SimParams params) {
    int lx = threadIdx.x;
    int ly = threadIdx.y;
    int x = start_x + lx;
    int y = start_y + ly;
    if (x >= nx || y >= ny) {
        return;
    }
    int idx = y * nx + x;
    float fx = (nx > 1) ? (-1.0f + 2.0f * x / (nx - 1)) : 0.0f;
    float fy = (ny > 1) ? (-1.0f + 2.0f * y / (ny - 1)) : 0.0f;
    float bx = 0.5f * params.separation * __cosf(params.omega_orbit * t);
    float by = 0.5f * params.separation * __sinf(params.omega_orbit * t);
    float bx2 = -bx;
    float by2 = -by;
    float dx1 = fx - bx;
    float dy1 = fy - by;
    float dx2 = fx - bx2;
    float dy2 = fy - by2;
    float r1 = sqrtf(dx1 * dx1 + dy1 * dy1);
    float r2 = sqrtf(dx2 * dx2 + dy2 * dy2);

    float swirl = 0.0f;
    if (r1 < params.refine_radius || r2 < params.refine_radius) {
        float theta = atan2f(fy, fx);
        swirl = params.refine_amp * __sinf(8.0f * theta + 2.0f * t)
                * expf(-20.0f * (r1 * r1 + r2 * r2));
        rho[idx] += swirl;
        phi[idx] += 0.2f * swirl;
    }
}

__global__ __launch_bounds__(256)
void update_binary_flow(
    float *rho,
    float *phi,
    int nx,
    int ny,
    const float *time_ptr,
    SimParams params) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }
    float t = time_ptr[0];
    int idx = y * nx + x;
    float fx = (nx > 1) ? (-1.0f + 2.0f * x / (nx - 1)) : 0.0f;
    float fy = (ny > 1) ? (-1.0f + 2.0f * y / (ny - 1)) : 0.0f;
    float r = sqrtf(fx * fx + fy * fy);
    float theta = atan2f(fy, fx);

    float bx = 0.5f * params.separation * __cosf(params.omega_orbit * t);
    float by = 0.5f * params.separation * __sinf(params.omega_orbit * t);
    float bx2 = -bx;
    float by2 = -by;

    float dx1 = fx - bx;
    float dy1 = fy - by;
    float dx2 = fx - bx2;
    float dy2 = fy - by2;

    float r1 = safe_radius(sqrtf(dx1 * dx1 + dy1 * dy1), params.rs1);
    float r2 = safe_radius(sqrtf(dx2 * dx2 + dy2 * dy2), params.rs2);

    float phi_val = -params.gm1 / (r1 - params.rs1) - params.gm2 / (r2 - params.rs2);

    float disk = params.rho0 * expf(-((r - params.r0) * (r - params.r0)) / (2.0f * params.sigma * params.sigma));
    float spiral = 1.0f + params.arm_amp * __sinf(params.arm_m * (theta - params.omega_disk * t));
    float binary = 1.0f + params.bin_amp * __cosf(2.0f * theta - params.omega_orbit * t);
    float rho_val = disk * spiral * binary;

    if (r < params.rs1 || r < params.rs2) {
        rho_val = 0.0f;
    }

    rho[idx] = rho_val;
    phi[idx] = phi_val;

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        float cx = (nx > 1) ? (-1.0f + 2.0f * (blockIdx.x * blockDim.x + blockDim.x / 2) / (nx - 1)) : 0.0f;
        float cy = (ny > 1) ? (-1.0f + 2.0f * (blockIdx.y * blockDim.y + blockDim.y / 2) / (ny - 1)) : 0.0f;
        float d1 = sqrtf((cx - bx) * (cx - bx) + (cy - by) * (cy - by));
        float d2 = sqrtf((cx - bx2) * (cx - bx2) + (cy - by2) * (cy - by2));
        if (d1 < params.refine_radius || d2 < params.refine_radius) {
            dim3 refine_block(blockDim.x, blockDim.y, 1);
            refine_patch<<<1, refine_block>>>(
                rho, phi, nx, ny,
                blockIdx.x * blockDim.x,
                blockIdx.y * blockDim.y,
                t, params);
        }
    }
}

__global__ void real_to_complex(const float *in, cufftComplex *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx].x = in[idx];
        out[idx].y = 0.0f;
    }
}

__global__ void complex_to_real(const cufftComplex *in, float *out, int n, float scale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx].x * scale;
    }
}

__global__ void spectral_filter(cufftComplex *freq, int nx, int ny, float cutoff) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }
    int idx = y * nx + x;
    int kx = (x <= nx / 2) ? x : x - nx;
    int ky = (y <= ny / 2) ? y : y - ny;
    float k2 = static_cast<float>(kx * kx + ky * ky);
    float c2 = cutoff * cutoff;
    float scale = expf(-k2 / c2);
    freq[idx].x *= scale;
    freq[idx].y *= scale;
}

static void write_frame(const fs::path &path, const float *data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
}

int main(int argc, char **argv) {
    int nx = 192;
    int ny = 192;
    int frames = 1440;
    float dt = 0.05f;
    std::string out_dir = "frames_binary";

    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--nx") == 0) {
            nx = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--ny") == 0) {
            ny = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--frames") == 0) {
            frames = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--dt") == 0) {
            dt = std::strtof(argv[i + 1], nullptr);
        } else if (std::strcmp(argv[i], "--out") == 0) {
            out_dir = argv[i + 1];
        }
    }

    fs::create_directories(out_dir);

    int total = nx * ny;
    size_t bytes = static_cast<size_t>(total) * sizeof(float);

    float *d_rho = nullptr;
    float *d_phi = nullptr;
    float *d_noise = nullptr;
    float *d_time = nullptr;
    cufftComplex *d_phi_complex = nullptr;
    curandStatePhilox4_32_10_t *d_states = nullptr;

    check_cuda(cudaMalloc(&d_rho, bytes), "cudaMalloc rho");
    check_cuda(cudaMalloc(&d_phi, bytes), "cudaMalloc phi");
    check_cuda(cudaMalloc(&d_noise, bytes), "cudaMalloc noise");
    check_cuda(cudaMalloc(&d_time, sizeof(float)), "cudaMalloc time");
    check_cuda(cudaMalloc(&d_phi_complex, sizeof(cufftComplex) * total), "cudaMalloc phi_complex");
    check_cuda(cudaMalloc(&d_states, sizeof(curandStatePhilox4_32_10_t) * total), "cudaMalloc states");

    float *h_rho = nullptr;
    float *h_phi = nullptr;
    check_cuda(cudaMallocHost(&h_rho, bytes), "cudaMallocHost rho");
    check_cuda(cudaMallocHost(&h_phi, bytes), "cudaMallocHost phi");

    cudaStream_t stream;
    check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

    check_cuda(cudaDeviceSetLimit(cudaLimitDevRuntimePendingLaunchCount, 2048), "set pending launch");
    check_cuda(cudaDeviceSetLimit(cudaLimitDevRuntimeSyncDepth, 2), "set sync depth");

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    setup_curand<<<blocks, threads, 0, stream>>>(d_states, 1234ULL, total);
    check_cuda(cudaGetLastError(), "setup_curand");

    cufftHandle plan;
    check_cufft(cufftPlan2d(&plan, ny, nx, CUFFT_C2C), "cufftPlan2d");
    check_cufft(cufftSetStream(plan, stream), "cufftSetStream");

    cublasHandle_t cublas;
    check_cublas(cublasCreate(&cublas), "cublasCreate");
    check_cublas(cublasSetStream(cublas, stream), "cublasSetStream");
    check_cublas(cublasSetPointerMode(cublas, CUBLAS_POINTER_MODE_HOST), "cublas pointer mode");

    SimParams params;
    params.gm1 = 1.0f;
    params.gm2 = 0.8f;
    params.rs1 = 0.12f;
    params.rs2 = 0.1f;
    params.separation = 0.5f;
    params.omega_orbit = 1.2f;
    params.omega_disk = 2.2f;
    params.rho0 = 1.0f;
    params.r0 = 0.55f;
    params.sigma = 0.15f;
    params.arm_amp = 0.35f;
    params.arm_m = 3.0f;
    params.bin_amp = 0.2f;
    params.refine_radius = 0.35f;
    params.refine_amp = 0.08f;
    params.filter_cutoff = 20.0f;

    float noise_scale = 0.02f;

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    check_cuda(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal), "begin capture");

    update_binary_flow<<<grid, block, 0, stream>>>(d_rho, d_phi, nx, ny, d_time, params);
    noise_kernel<<<blocks, threads, 0, stream>>>(d_states, d_noise, total);
    check_cublas(cublasSaxpy(cublas, total, &noise_scale, d_noise, 1, d_rho, 1), "cublasSaxpy");

    real_to_complex<<<blocks, threads, 0, stream>>>(d_phi, d_phi_complex, total);
    check_cufft(cufftExecC2C(plan, d_phi_complex, d_phi_complex, CUFFT_FORWARD), "cufft forward");
    spectral_filter<<<grid, block, 0, stream>>>(d_phi_complex, nx, ny, params.filter_cutoff);
    check_cufft(cufftExecC2C(plan, d_phi_complex, d_phi_complex, CUFFT_INVERSE), "cufft inverse");
    complex_to_real<<<blocks, threads, 0, stream>>>(d_phi_complex, d_phi, total, 1.0f / total);

    check_cuda(cudaStreamEndCapture(stream, &graph), "end capture");
    check_cuda(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0), "graph instantiate");

    for (int frame = 0; frame < frames; ++frame) {
        float t = frame * dt;
        check_cuda(cudaMemcpyAsync(d_time, &t, sizeof(float), cudaMemcpyHostToDevice, stream), "memcpy time");
        check_cuda(cudaGraphLaunch(graph_exec, stream), "graph launch");
        check_cuda(cudaMemcpyAsync(h_rho, d_rho, bytes, cudaMemcpyDeviceToHost, stream), "memcpy rho");
        check_cuda(cudaMemcpyAsync(h_phi, d_phi, bytes, cudaMemcpyDeviceToHost, stream), "memcpy phi");
        check_cuda(cudaStreamSynchronize(stream), "stream sync");

        char name[64];
        std::snprintf(name, sizeof(name), "rho_%04d.bin", frame);
        write_frame(fs::path(out_dir) / name, h_rho, bytes);
        std::snprintf(name, sizeof(name), "phi_%04d.bin", frame);
        write_frame(fs::path(out_dir) / name, h_phi, bytes);

        if (frame % 200 == 0) {
            std::printf("Generated frame %d/%d\n", frame, frames);
        }
    }

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cublasDestroy(cublas);
    cufftDestroy(plan);

    cudaFreeHost(h_rho);
    cudaFreeHost(h_phi);

    cudaFree(d_rho);
    cudaFree(d_phi);
    cudaFree(d_noise);
    cudaFree(d_time);
    cudaFree(d_phi_complex);
    cudaFree(d_states);
    cudaStreamDestroy(stream);

    return 0;
}
