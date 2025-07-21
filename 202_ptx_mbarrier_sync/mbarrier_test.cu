#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void mbarrier_kernel(int *out) {
#if __CUDA_ARCH__ >= 900
    __shared__ unsigned long long barrier;
    if (threadIdx.x == 0) {
        asm volatile("mbarrier.init.shared.b64 [%0], %1;" :: "l"(&barrier), "r"(blockDim.x));
    }
    __syncthreads();

    asm volatile("mbarrier.arrive.shared.b64 %0, [%1];" :: "l"(0ULL), "l"(&barrier));
    asm volatile("mbarrier.wait.shared.b64 %0, [%1];" :: "l"(0ULL), "l"(&barrier));

    if (threadIdx.x == 0) {
        out[0] = 1;
    }
#else
    if (threadIdx.x == 0) {
        out[0] = 0;
    }
#endif
}

int main() {
    int *d_out = nullptr;
    cudaMalloc(&d_out, sizeof(int));
    mbarrier_kernel<<<1, 128>>>(d_out);
    cudaDeviceSynchronize();

    int h = -1;
    cudaMemcpy(&h, d_out, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_out);

    std::printf("mbarrier result=%d\n", h);
    return 0;
}
