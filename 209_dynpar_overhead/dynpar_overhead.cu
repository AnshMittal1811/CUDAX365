#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void refine_patch(float *grid, int nx, int ny) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny;
    if (idx < total) {
        grid[idx] *= 1.001f;
    }
}

__global__ void dynpar_kernel(float *grid, int nx, int ny) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        dim3 block(256);
        dim3 grid_dim((nx * ny + block.x - 1) / block.x);
        refine_patch<<<grid_dim, block>>>(grid, nx, ny);
    }
}

static float time_kernel(void (*kernel)(float *, int, int), float *grid, int nx, int ny) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    kernel<<<1, 1>>>(grid, nx, ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

static float time_host(float *grid, int nx, int ny) {
    dim3 block(256);
    dim3 grid_dim((nx * ny + block.x - 1) / block.x);
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    refine_patch<<<grid_dim, block>>>(grid, nx, ny);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}

int main(int argc, char **argv) {
    int nx = (argc > 1) ? std::atoi(argv[1]) : 256;
    int ny = (argc > 2) ? std::atoi(argv[2]) : 256;
    size_t total = static_cast<size_t>(nx) * ny;

    float *grid = nullptr;
    cudaMalloc(&grid, total * sizeof(float));

    float dynpar_ms = time_kernel(dynpar_kernel, grid, nx, ny);
    float host_ms = time_host(grid, nx, ny);

    std::printf("dynpar_ms=%.4f host_ms=%.4f\n", dynpar_ms, host_ms);

    cudaFree(grid);
    return 0;
}
