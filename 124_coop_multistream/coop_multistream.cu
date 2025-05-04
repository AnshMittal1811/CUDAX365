#include <cuda_runtime.h>
#include <cstdio>

__global__ void coop_kernel(float* data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    data[i] = data[i] + 1.0f;
}

int main(){
    int n = 1 << 20;
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);

    void* args[] = {&d};
    int block = 256;
    int grid = (n + block - 1) / block;

    cudaLaunchCooperativeKernel((void*)coop_kernel, grid, block, args, 0, s1);
    cudaLaunchCooperativeKernel((void*)coop_kernel, grid, block, args, 0, s2);

    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    printf("coop kernels done\n");

    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFree(d);
    return 0;
}
