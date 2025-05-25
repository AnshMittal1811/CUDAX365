#include <cuda_runtime.h>
#include <mma.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

using namespace nvcuda;

static void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

__device__ __forceinline__ float fma_ptx(float a, float b, float c) {
    float out;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}

__global__ void cross_warp_kernel(const float *a, const float *b, float *c, int matrices) {
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread / warpSize;
    int lane = threadIdx.x & 31;
    if (warp_id >= matrices) {
        return;
    }
    if (lane >= 9) {
        return;
    }

    int base = warp_id * 9;
    int row = lane / 3;
    int col = lane % 3;
    int row_base = row * 3;

    float a_val = a[base + row_base + col];
    float b_val = b[base + row_base + col];

    unsigned mask = 0x1FF;
    float ax = __shfl_sync(mask, a_val, row_base + 0);
    float ay = __shfl_sync(mask, a_val, row_base + 1);
    float az = __shfl_sync(mask, a_val, row_base + 2);
    float bx = __shfl_sync(mask, b_val, row_base + 0);
    float by = __shfl_sync(mask, b_val, row_base + 1);
    float bz = __shfl_sync(mask, b_val, row_base + 2);

    float out = 0.0f;
    if (col == 0) {
        out = fma_ptx(ay, bz, -az * by);
    } else if (col == 1) {
        out = fma_ptx(az, bx, -ax * bz);
    } else {
        out = fma_ptx(ax, by, -ay * bx);
    }

    c[base + row_base + col] = out;
}

__global__ void wmma_gemm_kernel(const half *a, const half *b, float *c, int tiles) {
#if __CUDA_ARCH__ >= 700
    int tile = blockIdx.x;
    if (tile >= tiles) {
        return;
    }

    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);
    const half *a_tile = a + tile * 16 * 16;
    const half *b_tile = b + tile * 16 * 16;
    float *c_tile = c + tile * 16 * 16;

    wmma::load_matrix_sync(a_frag, a_tile, 16);
    wmma::load_matrix_sync(b_frag, b_tile, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(c_tile, c_frag, 16, wmma::mem_row_major);
#else
    (void)a;
    (void)b;
    (void)c;
    (void)tiles;
#endif
}

static float time_cross(const float *a, const float *b, float *c, int matrices, int threads) {
    int warps_per_block = threads / 32;
    int blocks = (matrices + warps_per_block - 1) / warps_per_block;
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event start");
    check(cudaEventCreate(&stop), "event stop");
    check(cudaEventRecord(start), "record start");
    cross_warp_kernel<<<blocks, threads>>>(a, b, c, matrices);
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "sync stop");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    check(cudaEventDestroy(start), "destroy start");
    check(cudaEventDestroy(stop), "destroy stop");
    return ms;
}

static float time_wmma(const half *a, const half *b, float *c, int tiles) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event start");
    check(cudaEventCreate(&stop), "event stop");
    check(cudaEventRecord(start), "record start");
    wmma_gemm_kernel<<<tiles, 32>>>(a, b, c, tiles);
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "sync stop");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    check(cudaEventDestroy(start), "destroy start");
    check(cudaEventDestroy(stop), "destroy stop");
    return ms;
}

int main(int argc, char **argv) {
    int matrices = (argc > 1) ? std::atoi(argv[1]) : 8192;
    int tiles = (argc > 2) ? std::atoi(argv[2]) : 1024;

    size_t total = static_cast<size_t>(matrices) * 9;
    std::vector<float> h_a(total, 0.25f);
    std::vector<float> h_b(total, 0.5f);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    check(cudaMalloc(&d_a, total * sizeof(float)), "malloc d_a");
    check(cudaMalloc(&d_b, total * sizeof(float)), "malloc d_b");
    check(cudaMalloc(&d_c, total * sizeof(float)), "malloc d_c");
    check(cudaMemcpy(d_a, h_a.data(), total * sizeof(float), cudaMemcpyHostToDevice), "H2D a");
    check(cudaMemcpy(d_b, h_b.data(), total * sizeof(float), cudaMemcpyHostToDevice), "H2D b");

    size_t tile_elems = static_cast<size_t>(tiles) * 16 * 16;
    half *d_ha = nullptr;
    half *d_hb = nullptr;
    float *d_hc = nullptr;
    check(cudaMalloc(&d_ha, tile_elems * sizeof(half)), "malloc d_ha");
    check(cudaMalloc(&d_hb, tile_elems * sizeof(half)), "malloc d_hb");
    check(cudaMalloc(&d_hc, tile_elems * sizeof(float)), "malloc d_hc");

    float ms_cross = time_cross(d_a, d_b, d_c, matrices, 256);
    float ms_wmma = time_wmma(d_ha, d_hb, d_hc, tiles);

    std::printf("cross_ms=%.4f wmma_ms=%.4f\n", ms_cross, ms_wmma);

    check(cudaFree(d_a), "free d_a");
    check(cudaFree(d_b), "free d_b");
    check(cudaFree(d_c), "free d_c");
    check(cudaFree(d_ha), "free d_ha");
    check(cudaFree(d_hb), "free d_hb");
    check(cudaFree(d_hc), "free d_hc");

    return 0;
}
