#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void bench_kernel(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float v = static_cast<float>(idx) * 0.001f;
    #pragma unroll 4
    for (int i = 0; i < 256; ++i) {
        v = v * 1.0001f + 0.0003f;
        v = v - 0.0002f;
    }
    out[idx] = v;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 22);
    float *d_out = nullptr;
    cudaMalloc(&d_out, n * sizeof(float));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    bench_kernel<<<blocks, threads>>>(d_out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::printf("bench_kernel_ms=%.4f\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_out);
    return 0;
}
