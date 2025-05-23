#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

__global__ void kernel_strided(const float *in, float *out, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx * stride;
    if (i < n) {
        out[i] = in[i] * 1.001f + 0.1f;
    }
}

__global__ void kernel_coalesced(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx] * 1.001f + 0.1f;
    }
}

static float time_kernel(void (*kernel)(const float *, float *, int, int),
                         const float *in, float *out, int n, int stride,
                         int grid, int block) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");
    check(cudaEventRecord(start), "event record start");
    kernel<<<grid, block>>>(in, out, n, stride);
    check(cudaEventRecord(stop), "event record stop");
    check(cudaEventSynchronize(stop), "event sync stop");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    check(cudaEventDestroy(start), "event destroy start");
    check(cudaEventDestroy(stop), "event destroy stop");
    return ms;
}

static float time_kernel2(void (*kernel)(const float *, float *, int),
                          const float *in, float *out, int n,
                          int grid, int block) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");
    check(cudaEventRecord(start), "event record start");
    kernel<<<grid, block>>>(in, out, n);
    check(cudaEventRecord(stop), "event record stop");
    check(cudaEventSynchronize(stop), "event sync stop");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    check(cudaEventDestroy(start), "event destroy start");
    check(cudaEventDestroy(stop), "event destroy stop");
    return ms;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);
    int block = (argc > 2) ? std::atoi(argv[2]) : 256;
    int stride = (argc > 3) ? std::atoi(argv[3]) : 4;
    if (n <= 0 || block <= 0 || stride <= 0) {
        std::fprintf(stderr, "n, block, stride must be positive\n");
        return 1;
    }

    float *d_in = nullptr;
    float *d_out = nullptr;
    check(cudaMalloc(&d_in, n * sizeof(float)), "cudaMalloc d_in");
    check(cudaMalloc(&d_out, n * sizeof(float)), "cudaMalloc d_out");

    int grid = (n + block - 1) / block;
    float ms_strided = time_kernel(kernel_strided, d_in, d_out, n, stride, grid, block);
    float ms_coalesced = time_kernel2(kernel_coalesced, d_in, d_out, n, grid, block);

    std::printf("strided_ms=%.4f coalesced_ms=%.4f stride=%d\n",
                ms_strided, ms_coalesced, stride);

    check(cudaFree(d_in), "cudaFree d_in");
    check(cudaFree(d_out), "cudaFree d_out");
    return 0;
}
