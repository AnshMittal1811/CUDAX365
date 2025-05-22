#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

__device__ __forceinline__ float mix_ops(float x) {
    x = x * 1.0001f + 0.0003f;
    x = x - 0.0002f;
    x = x * 0.9999f + 0.0001f;
    return x;
}

__global__ void kernel_baseline(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float a = in[idx];
    float b = a + 1.0f;
    float c = b * 1.01f;
    float d = c - 0.33f;
    float e = d * 0.99f;
    float f = e + 0.25f;
    out[idx] = mix_ops(f);
}

__global__ void kernel_reordered(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float a = in[idx];
    float b = a + 1.0f;
    float d = a - 0.33f;
    float c = b * 1.01f;
    float e = d * 0.99f;
    float f = e + 0.25f;
    out[idx] = mix_ops(c + f);
}

static float run_kernel(void (*kernel)(const float *, float *, int), float *d_in, float *d_out,
                        int n, int block) {
    int grid = (n + block - 1) / block;
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");

    check(cudaEventRecord(start), "event record start");
    kernel<<<grid, block>>>(d_in, d_out, n);
    check(cudaEventRecord(stop), "event record stop");
    check(cudaEventSynchronize(stop), "event sync stop");

    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    check(cudaEventDestroy(start), "event destroy start");
    check(cudaEventDestroy(stop), "event destroy stop");
    return ms;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 22);
    int block = (argc > 2) ? std::atoi(argv[2]) : 256;
    if (n <= 0 || block <= 0) {
        std::fprintf(stderr, "n and block must be positive\n");
        return 1;
    }

    float *d_in = nullptr;
    float *d_out = nullptr;
    check(cudaMalloc(&d_in, n * sizeof(float)), "cudaMalloc d_in");
    check(cudaMalloc(&d_out, n * sizeof(float)), "cudaMalloc d_out");

    float ms_base = run_kernel(kernel_baseline, d_in, d_out, n, block);
    float ms_reorder = run_kernel(kernel_reordered, d_in, d_out, n, block);

    std::printf("baseline_ms=%.4f reordered_ms=%.4f\n", ms_base, ms_reorder);

    check(cudaFree(d_in), "cudaFree d_in");
    check(cudaFree(d_out), "cudaFree d_out");
    return 0;
}
