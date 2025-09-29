#include <cstdio>

__global__ void pde_kernel(float *grid, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        grid[idx] = grid[idx] * 0.99f + 0.01f;
    }
}

int main() {
    std::printf("cuda pde kernel placeholder\n");
    return 0;
}
