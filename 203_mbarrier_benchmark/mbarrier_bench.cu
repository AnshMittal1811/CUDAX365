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

__global__ void sync_kernel(int *out) {
    __syncthreads();
    if (threadIdx.x == 0) {
        out[0] = 1;
    }
}

static float time_kernel(void (*kernel)(int *), int *d_out, int iters) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    for (int i = 0; i < iters; ++i) {
        kernel<<<1, 128>>>(d_out);
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
    int *d_out = nullptr;
    cudaMalloc(&d_out, sizeof(int));

    float mbarrier_ms = time_kernel(mbarrier_kernel, d_out, iters);
    float sync_ms = time_kernel(sync_kernel, d_out, iters);

    std::printf("mbarrier_ms=%.6f sync_ms=%.6f\n", mbarrier_ms, sync_ms);

    cudaFree(d_out);
    return 0;
}
