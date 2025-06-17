#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void stress_kernel(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float v = static_cast<float>(idx) * 0.001f;
    #pragma unroll 8
    for (int i = 0; i < 512; ++i) {
        v = v * 1.0001f + 0.0003f;
        v = v - 0.0002f;
    }
    out[idx] = v;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);
    float *d_out = nullptr;
    cudaMalloc(&d_out, n * sizeof(float));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    stress_kernel<<<blocks, threads>>>(d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_out);
    std::printf("stress kernel done\n");
    return 0;
}
