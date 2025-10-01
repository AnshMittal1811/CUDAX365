#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void halo_kernel(const float *in, float *out, int n) {
    extern __shared__ float sh[];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
#if __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;" :: "r"(sh + threadIdx.x), "l"(in + idx), "n"(4));
    asm volatile("cp.async.commit_group;" ::);
    asm volatile("cp.async.wait_group 0;" ::);
    __syncthreads();
    out[idx] = sh[threadIdx.x];
#else
    out[idx] = in[idx];
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
    halo_kernel<<<blocks, threads, threads * sizeof(float)>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    std::printf("cp.async halo done\n");
    return 0;
}
