#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ float add_ptx(float a, float b) {
    float out;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__global__ void reward_shaping(const float *r1, const float *r2, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = add_ptx(r1[idx], r2[idx]);
        out[idx] = v > 0.0f ? v : 0.0f;
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 1024;
    float *d_r1 = nullptr;
    float *d_r2 = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_r1, n * sizeof(float));
    cudaMalloc(&d_r2, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    reward_shaping<<<(n + 255) / 256, 256>>>(d_r1, d_r2, d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_r1);
    cudaFree(d_r2);
    cudaFree(d_out);
    std::printf("reward shaping done\n");
    return 0;
}
