#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void baseline_kernel(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = in[idx];
        float b = a * 1.01f;
        float c = b + 0.5f;
        float d = c * 0.99f;
        out[idx] = d + 0.1f;
    }
}

__global__ void reordered_kernel(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float a = in[idx];
        float c = a + 0.5f;
        float b = a * 1.01f;
        float d = b * 0.99f;
        out[idx] = d + c * 0.1f;
    }
}

static float time_kernel(void (*kernel)(const float *, float *, int), const float *in, float *out, int n) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<(n + 255) / 256, 256>>>(in, out, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    float *d_in = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    float base_ms = time_kernel(baseline_kernel, d_in, d_out, n);
    float reorder_ms = time_kernel(reordered_kernel, d_in, d_out, n);

    std::printf("baseline_ms=%.6f reordered_ms=%.6f\n", base_ms, reorder_ms);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
