#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel(float* data, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) data[i] = data[i] * 1.01f + 0.1f;
}

int main(int argc, char** argv){
    int n = 1 << 20;
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));
    int block = (argc > 1) ? atoi(argv[1]) : 256;
    int grid = (n + block - 1) / block;
    for (int i=0;i<100;i++) kernel<<<grid, block>>>(d, n);
    cudaDeviceSynchronize();
    cudaFree(d);
    return 0;
}
