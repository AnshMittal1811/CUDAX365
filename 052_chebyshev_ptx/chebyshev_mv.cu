#include <cuda_runtime.h>
#include <cstdio>

__device__ __forceinline__ float fma_rn(float a, float b, float c){
    float out;
    asm volatile("fma.rn.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}

__global__ void cheb_mv(const float* A, const float* x, float* y, int n){
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n) return;
    float sum = 0.0f;
    for (int k=0;k<n;k++) sum = fma_rn(A[row*n + k], x[k], sum);
    y[row] = sum;
}

int main(){
    int n = 64;
    float *dA=nullptr, *dx=nullptr, *dy=nullptr;
    cudaMalloc(&dA, n*n*sizeof(float));
    cudaMalloc(&dx, n*sizeof(float));
    cudaMalloc(&dy, n*sizeof(float));
    cudaMemset(dA, 0, n*n*sizeof(float));
    cudaMemset(dx, 0, n*sizeof(float));
    cheb_mv<<<(n+255)/256, 256>>>(dA, dx, dy, n);
    cudaDeviceSynchronize();
    printf("cheb mv done\n");
    cudaFree(dA);
    cudaFree(dx);
    cudaFree(dy);
    return 0;
}
