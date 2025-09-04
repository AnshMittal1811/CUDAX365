#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void syncwarp_kernel(float *out) {
    __syncwarp();
    if (threadIdx.x == 0) {
        out[0] = 1.0f;
    }
}

__global__ void barsync_kernel(float *out) {
    asm volatile("bar.sync 0;" ::: "memory");
    if (threadIdx.x == 0) {
        out[0] = 1.0f;
    }
}

static float time_kernel(void (*kernel)(float *), float *out, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        kernel<<<1, 32>>>(out);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms / iters;
}

int main(int argc, char **argv) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : 1000;
    float *d_out = nullptr;
    cudaMalloc(&d_out, sizeof(float));

    float syncwarp_ms = time_kernel(syncwarp_kernel, d_out, iters);
    float barsync_ms = time_kernel(barsync_kernel, d_out, iters);

    std::printf("syncwarp_ms=%.6f barsync_ms=%.6f\n", syncwarp_ms, barsync_ms);

    cudaFree(d_out);
    return 0;
}
