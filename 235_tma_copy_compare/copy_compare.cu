#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void copy_kernel(const float *in, float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
}

__global__ void cp_async_kernel(const float *in, float *out, int n) {
#if __CUDA_ARCH__ >= 900
    extern __shared__ float sh[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        asm volatile("cp.async.bulk.shared.global [%0], [%1], %2;" :: "r"(sh + threadIdx.x), "l"(in + idx), "n"(4));
    }
    __syncthreads();
    if (idx < n) {
        out[idx] = sh[threadIdx.x];
    }
#else
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
#endif
}

static float time_kernel(void (*kernel)(const float *, float *, int), const float *in, float *out, int n, int shmem) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<(n + 255) / 256, 256, shmem>>>(in, out, n);
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

    float ms_copy = time_kernel(copy_kernel, d_in, d_out, n, 0);
    float ms_tma = time_kernel(cp_async_kernel, d_in, d_out, n, 256 * sizeof(float));

    std::printf("copy_ms=%.4f tma_ms=%.4f\n", ms_copy, ms_tma);

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
