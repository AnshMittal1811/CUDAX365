#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static float time_copy(cudaMemcpyKind kind, float *dst, float *src, size_t bytes) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    cudaMemcpy(dst, src, bytes, kind);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main(int argc, char **argv) {
    size_t bytes = (argc > 1) ? static_cast<size_t>(std::atoll(argv[1])) : (1 << 24);

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *h = nullptr;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMallocHost(&h, bytes);

    float ms_dev = time_copy(cudaMemcpyDeviceToDevice, d_b, d_a, bytes);
    float ms_h2d = time_copy(cudaMemcpyDeviceToHost, h, d_a, bytes);
    float ms_d2h = time_copy(cudaMemcpyHostToDevice, d_b, h, bytes);

    std::printf("dev_to_dev_ms=%.4f host_stage_ms=%.4f\n", ms_dev, ms_h2d + ms_d2h);

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFreeHost(h);
    return 0;
}
