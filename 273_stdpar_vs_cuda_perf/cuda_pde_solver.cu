#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void pde_kernel(float *grid, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grid[idx] = grid[idx] * 0.99f + 0.01f;
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    float *d_grid = nullptr;
    cudaMalloc(&d_grid, n * sizeof(float));

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    pde_kernel<<<(n + 255) / 256, 256>>>(d_grid, n);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    std::printf("cuda_pde_ms=%.4f\n", ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_grid);
    return 0;
}
