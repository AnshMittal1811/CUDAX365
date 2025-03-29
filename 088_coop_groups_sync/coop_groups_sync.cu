#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cstdio>

namespace cg = cooperative_groups;

__global__ void kernel_sync(float* data){
    cg::thread_block block = cg::this_thread_block();
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    data[i] = data[i] + 1.0f;
    block.sync();
    data[i] = data[i] + 1.0f;
}

__global__ void kernel_syncthreads(float* data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    data[i] = data[i] + 1.0f;
    __syncthreads();
    data[i] = data[i] + 1.0f;
}

int main(){
    int n = 1 << 20;
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));
    kernel_sync<<<(n+255)/256, 256>>>(d);
    kernel_syncthreads<<<(n+255)/256, 256>>>(d);
    cudaDeviceSynchronize();
    printf("sync kernels done\n");
    cudaFree(d);
    return 0;
}
