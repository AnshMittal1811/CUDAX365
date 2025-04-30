#include <cuda_runtime.h>
#include <cstdio>

__global__ __launch_bounds__(256, 2) void conv1d(const float* in, float* out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= 0 || i >= n-1) return;
    out[i] = 0.25f * in[i-1] + 0.5f * in[i] + 0.25f * in[i+1];
}

int main(){
    int n = 1 << 20;
    float *d_in=nullptr, *d_out=nullptr;
    cudaMalloc(&d_in, n*sizeof(float));
    cudaMalloc(&d_out, n*sizeof(float));
    cudaMemset(d_in, 0, n*sizeof(float));
    conv1d<<<(n+255)/256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    printf("conv done\n");
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
