#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void mbarrier_copy(const float *in, float *out, int n) {
#if __CUDA_ARCH__ >= 900
    __shared__ unsigned long long barrier;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "l"(&barrier), "r"(blockDim.x));
    }
    __syncthreads();

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }

    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" :: "l"(0ULL), "l"(&barrier));
    asm volatile("mbarrier.wait.shared.b64 %0, [%1];" :: "l"(0ULL), "l"(&barrier));
#else
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = in[idx];
    }
#endif
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 1024;
    float *d_in = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));

    mbarrier_copy<<<(n + 255) / 256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    std::printf("mbarrier async copy done\n");
    return 0;
}
