#include <cuda_runtime.h>
#include <cstdio>

__device__ __forceinline__ int shfl_down(int v, int delta){
    int out;
    asm volatile("shfl.down.b32 %0, %1, %2, 0xffffffff;" : "=r"(out) : "r"(v), "r"(delta));
    return out;
}

__global__ void scatter_add(const float* in, float* out, int n){
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= n) return;
    float v = in[tid];
    for (int offset=16; offset>0; offset/=2) v += shfl_down(v, offset);
    if ((threadIdx.x & 31) == 0) atomicAdd(&out[blockIdx.x], v);
}

int main(){
    int n = 1 << 20;
    float *d_in=nullptr, *d_out=nullptr;
    cudaMalloc(&d_in, n*sizeof(float));
    cudaMalloc(&d_out, 4096*sizeof(float));
    cudaMemset(d_in, 0, n*sizeof(float));
    cudaMemset(d_out, 0, 4096*sizeof(float));
    scatter_add<<<(n+255)/256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    printf("scatter done\n");
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
