#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void fourier_kernel(const float *in, float *out, int n) {
#if __CUDA_ARCH__ >= 900
    extern __shared__ float sh[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        asm volatile("cp.async.bulk.shared.global [%0], [%1], %2;" :: "r"(sh + threadIdx.x), "l"(in + idx), "n"(4));
    }
    __syncthreads();
    if (idx < n) {
        float v = sh[threadIdx.x];
        out[idx] = __sinf(v) + __cosf(v);
    }
#else
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = in[idx];
        out[idx] = __sinf(v) + __cosf(v);
    }
#endif
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 1024;
    float *d_in = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    fourier_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    std::printf("tma fourier lookup done\n");
    return 0;
}
