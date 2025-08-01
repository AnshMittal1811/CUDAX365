#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ float ld_global(const float *ptr) {
    float out;
    asm("ld.global.f32 %0, [%1];" : "=f"(out) : "l"(ptr));
    return out;
}

__global__ void depth_gather(const float *depth, const int *indices, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    int pos = indices[idx];
    out[idx] = ld_global(depth + pos);
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 1024;
    float *d_depth = nullptr;
    int *d_idx = nullptr;
    float *d_out = nullptr;

    cudaMalloc(&d_depth, n * sizeof(float));
    cudaMalloc(&d_idx, n * sizeof(int));
    cudaMalloc(&d_out, n * sizeof(float));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    depth_gather<<<blocks, threads>>>(d_depth, d_idx, d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_depth);
    cudaFree(d_idx);
    cudaFree(d_out);

    std::printf("depth_gather done\n");
    return 0;
}
