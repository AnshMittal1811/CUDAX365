#include <cuda_runtime.h>
#include <cstdio>

__global__ void flux_kernel(const float* u, float* out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= 0 || i >= n-1) return;
    out[i] = u[i] - 0.1f * (u[i+1] - u[i-1]);
}

int main(){
    int n = 1 << 20;
    float *d_u=nullptr, *d_out=nullptr;
    cudaMalloc(&d_u, n*sizeof(float));
    cudaMalloc(&d_out, n*sizeof(float));
    cudaMemset(d_u, 0, n*sizeof(float));
    flux_kernel<<<(n+255)/256, 256>>>(d_u, d_out, n);
    cudaDeviceSynchronize();
    printf("cuda done\n");
    cudaFree(d_u);
    cudaFree(d_out);
    return 0;
}
