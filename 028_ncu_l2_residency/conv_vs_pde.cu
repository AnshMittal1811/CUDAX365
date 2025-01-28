#include <cuda_runtime.h>
#include <cstdio>

__global__ void conv3x3(const float* in, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float acc = 0.0f;
    for (int k=-1;k<=1;k++){
        int j = idx + k;
        if (j >= 0 && j < n) acc += in[j] * 0.111f;
    }
    out[idx] = acc;
}

__global__ void pde_step(const float* u, float* out, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    float u0 = u[idx];
    float um1 = (idx > 0) ? u[idx-1] : u0;
    float up1 = (idx + 1 < n) ? u[idx+1] : u0;
    out[idx] = u0 + 0.1f * (um1 - 2.0f*u0 + up1);
}

int main(){
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    float *d_in=nullptr, *d_out=nullptr;
    cudaMalloc(&d_in, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemset(d_in, 0, bytes);
    int block = 256;
    int grid = (n + block - 1) / block;
    conv3x3<<<grid, block>>>(d_in, d_out, n);
    pde_step<<<grid, block>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    cudaFree(d_in);
    cudaFree(d_out);
    printf("done\n");
    return 0;
}
