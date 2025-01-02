// nvcc -O3 -arch=sm_89 vecadd.cu -o vecadd
#include <stdio.h>
#include <cuda_runtime.h>
#include <vector>
#include <random>
#include <cmath>

__global__ void vec_ad(const float* __restrict__ a,
                       const float* __restrict__ b,
                       float* __restrict__ c,
                       size_t n) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main(int argc, char** argv) {
    size_t N = (argc > 1) ? strtoull(argv[1], nullptr, 10): (size_t)1e8; // 100 million elements
    size_t bytes = N * sizeof(float);
    printf("N = %zu (%.2f MB per vector)\n", N, bytes / (1024.0 * 1024.0));

    //Host Init
    std::vector<float> h_a(N), h_b(N), h_c(N);
    std::mt19937 rng(123);
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    for (size_t i = 0; i < N; i++) {h_a[i] = dis(rng); h_b[i] = dis(rng);}

    //Device alloc
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);

    // Copy Host to Device
    cudaMemcpy(d_a, h_a.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b.data(), bytes, cudaMemcpyHostToDevice);  

    //Launch config for kernel
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;

    // warmup
    vec_ad<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaDeviceSynchronize();
    // Timed run with CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vec_ad<<<gridSize, blockSize>>>(d_a, d_b, d_c, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0.0f;
    cudaEventElapsedTime(&milliseconds, start, stop);   
    // Copy Device to Host and verify
    cudaMemcpy(h_c.data(), d_c, bytes, cudaMemcpyDeviceToHost);
    // Check a few elements
    double max_abs_err = 0.0;
    for (int i=0; i<10; i++) {
        size_t idx = (size_t)(N/10)*i;
        float ref = h_a[idx] + h_b[idx];
        max_abs_err = fmax(max_abs_err, fabs(h_c[idx] - ref));
    }

    // Effective bandwidth: 3 vectors read/written ~ 3 * bytes
    // Strictly: reads 2*bytes + writes 1*bytes = 3*bytes transferred by the kernel
    double gb = (3.0 * bytes) / (1e9);  // GB (decimal)
    double gbps = gb / (milliseconds / 1e3);

    printf("Time: %.3f ms  |  Effective BW: %.2f GB/s  |  max |err| (spot-check): %.3g\n",
           milliseconds, gbps, max_abs_err);

    // Cleanup
    cudaEventDestroy(start); cudaEventDestroy(stop);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    return 0;
}