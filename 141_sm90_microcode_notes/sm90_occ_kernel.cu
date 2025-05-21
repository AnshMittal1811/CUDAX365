#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

__global__ void busy_kernel(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float v = in[idx];
    #pragma unroll 8
    for (int i = 0; i < 64; ++i) {
        v = v * 1.00001f + 0.0001f;
        v = v - 0.00003f;
    }
    out[idx] = v;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    int block = (argc > 2) ? std::atoi(argv[2]) : 256;
    if (n <= 0 || block <= 0) {
        std::fprintf(stderr, "n and block must be positive\n");
        return 1;
    }

    float *d_in = nullptr;
    float *d_out = nullptr;
    check(cudaMalloc(&d_in, n * sizeof(float)), "cudaMalloc d_in");
    check(cudaMalloc(&d_out, n * sizeof(float)), "cudaMalloc d_out");

    int grid = (n + block - 1) / block;
    busy_kernel<<<grid, block>>>(d_in, d_out, n);
    check(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    check(cudaFree(d_in), "cudaFree d_in");
    check(cudaFree(d_out), "cudaFree d_out");

    std::printf("ran busy_kernel grid=%d block=%d\n", grid, block);
    return 0;
}
