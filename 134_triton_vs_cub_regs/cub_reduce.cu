#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdio>

int main(){
    int n = 1 << 20;
    float *d_in=nullptr, *d_out=nullptr;
    cudaMalloc(&d_in, n*sizeof(float));
    cudaMalloc(&d_out, sizeof(float));

    void* temp = nullptr;
    size_t temp_bytes = 0;
    cub::DeviceReduce::Sum(temp, temp_bytes, d_in, d_out, n);
    cudaMalloc(&temp, temp_bytes);
    cub::DeviceReduce::Sum(temp, temp_bytes, d_in, d_out, n);
    cudaDeviceSynchronize();
    printf("CUB reduce done\n");

    cudaFree(temp);
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
