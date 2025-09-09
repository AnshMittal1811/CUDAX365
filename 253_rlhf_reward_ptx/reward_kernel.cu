#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ float fma_ptx(float a, float b, float c) {
    float out;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}

__global__ void reward_kernel(const float *x, const float *w, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = fma_ptx(x[idx], w[idx], 0.1f);
        out[idx] = v > 0.0f ? v : 0.0f;
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 1024;
    float *d_x = nullptr;
    float *d_w = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_w, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    reward_kernel<<<(n + 255) / 256, 256>>>(d_x, d_w, d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_x);
    cudaFree(d_w);
    cudaFree(d_out);

    std::printf("reward kernel done\n");
    return 0;
}
