#include <cuda_runtime.h>
#include <cstdio>

__device__ __forceinline__ float shfl_down(float v, int delta){
    float out;
    asm volatile("shfl.down.b32 %0, %1, %2, 0xffffffff;" : "=f"(out) : "f"(v), "r"(delta));
    return out;
}

__global__ void lora_reduce(const float* grad, float* out, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    float v = (tid < n) ? grad[tid] : 0.0f;
    for (int offset=16; offset>0; offset/=2) v += shfl_down(v, offset);
    if ((threadIdx.x & 31) == 0) out[blockIdx.x * (blockDim.x/32) + threadIdx.x/32] = v;
}

int main(){
    int n = 1024;
    float *d_grad=nullptr, *d_out=nullptr;
    cudaMalloc(&d_grad, n*sizeof(float));
    cudaMalloc(&d_out, 32*sizeof(float));
    cudaMemset(d_grad, 0, n*sizeof(float));
    lora_reduce<<<1, 256>>>(d_grad, d_out, n);
    cudaDeviceSynchronize();
    printf("done\n");
    cudaFree(d_grad);
    cudaFree(d_out);
    return 0;
}
