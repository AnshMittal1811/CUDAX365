#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void refine_patch(float *grid, int nx, int ny, int start_x, int start_y, int size) {
    int x = start_x + blockIdx.x * blockDim.x + threadIdx.x;
    int y = start_y + blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= start_x + size || y >= start_y + size) {
        return;
    }
    grid[y * nx + x] *= 1.01f;
}

__global__ void amr_kernel(float *grid, int nx, int ny, float threshold) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny;
    if (idx >= total) {
        return;
    }
    float v = grid[idx];
    if (v > threshold && idx % 256 == 0) {
        int x = (idx % nx);
        int y = (idx / nx);
        dim3 block(8, 8);
        dim3 grid_dim(4, 4);
        refine_patch<<<grid_dim, block>>>(grid, nx, ny, x, y, 32);
    }
}

int main(int argc, char **argv) {
    int nx = (argc > 1) ? std::atoi(argv[1]) : 256;
    int ny = (argc > 2) ? std::atoi(argv[2]) : 256;

    size_t total = static_cast<size_t>(nx) * ny;
    float *d_grid = nullptr;
    cudaMalloc(&d_grid, total * sizeof(float));
    cudaMemset(d_grid, 0, total * sizeof(float));

    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    amr_kernel<<<blocks, threads>>>(d_grid, nx, ny, 0.5f);
    cudaDeviceSynchronize();

    cudaFree(d_grid);
    std::printf("dynpar amr done\n");
    return 0;
}
