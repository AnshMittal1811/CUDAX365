#include <cuda_runtime.h>
#include <cstdio>

__global__ __launch_bounds__(256, 2) void lb_kernel(float* data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    data[i] = data[i] * 2.0f;
}

int main(){
    int n = 1 << 20;
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));

    int blocks = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, lb_kernel, 256, 0);
    printf("max active blocks per SM: %d\n", blocks);

    lb_kernel<<<(n+255)/256, 256>>>(d);
    cudaDeviceSynchronize();
    cudaFree(d);
    return 0;
}
