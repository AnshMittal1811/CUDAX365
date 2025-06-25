#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>

__global__ void tiny_kernel(float *out) {
    if (threadIdx.x == 0) {
        out[0] += 1.0f;
    }
}

int main(int argc, char **argv) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : 10000;
    float *d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float));

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        tiny_kernel<<<1, 1>>>(d_out);
    }
    cudaDeviceSynchronize();
    auto end = std::chrono::high_resolution_clock::now();
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    double avg_us = static_cast<double>(us) / iters;
    std::printf("avg_launch_us=%.3f\n", avg_us);

    cudaFree(d_out);
    return 0;
}
