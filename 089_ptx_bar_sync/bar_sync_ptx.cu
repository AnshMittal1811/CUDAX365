#include <cuda_runtime.h>
#include <cstdio>

__global__ void bar_kernel(float* data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    data[i] = data[i] + 1.0f;
    asm volatile("bar.sync 0;");
    data[i] = data[i] + 1.0f;
}

int main(){
    int n = 1024;
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));
    bar_kernel<<<1, 256>>>(d);
    cudaDeviceSynchronize();
    printf("bar.sync done\n");
    cudaFree(d);
    return 0;
}
