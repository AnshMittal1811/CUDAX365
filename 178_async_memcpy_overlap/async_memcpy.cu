#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void scale_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 1.0001f;
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 22);

    float *h_data = nullptr;
    cudaHostAlloc(&h_data, n * sizeof(float), cudaHostAllocDefault);
    for (int i = 0; i < n; ++i) {
        h_data[i] = static_cast<float>(i) * 0.001f;
    }

    float *d_data = nullptr;
    cudaMalloc(&d_data, n * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    cudaMemcpyAsync(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice, stream);
    scale_kernel<<<(n + 255) / 256, 256, 0, stream>>>(d_data, n);
    cudaMemcpyAsync(h_data, d_data, n * sizeof(float), cudaMemcpyDeviceToHost, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::printf("async_copy_ms=%.4f\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    cudaFree(d_data);
    cudaFreeHost(h_data);
    return 0;
}
