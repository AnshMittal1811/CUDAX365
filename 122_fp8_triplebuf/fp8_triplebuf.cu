#include <cuda_runtime.h>
#include <cstdio>

__global__ void triple_buffer_kernel(const float* in, float* out, int n){
    __shared__ float buf0[256];
    __shared__ float buf1[256];
    __shared__ float buf2[256];

    int tid = threadIdx.x;
    int base = blockIdx.x * blockDim.x * 3;

    if (base + tid < n) buf0[tid] = in[base + tid];
    if (base + blockDim.x + tid < n) buf1[tid] = in[base + blockDim.x + tid];
    if (base + 2 * blockDim.x + tid < n) buf2[tid] = in[base + 2 * blockDim.x + tid];
    __syncthreads();

    if (base + tid < n) out[base + tid] = buf0[tid] * 1.01f;
    if (base + blockDim.x + tid < n) out[base + blockDim.x + tid] = buf1[tid] * 1.01f;
    if (base + 2 * blockDim.x + tid < n) out[base + 2 * blockDim.x + tid] = buf2[tid] * 1.01f;
}

int main(){
    int n = 1 << 20;
    float *d_in=nullptr, *d_out=nullptr;
    cudaMalloc(&d_in, n*sizeof(float));
    cudaMalloc(&d_out, n*sizeof(float));
    cudaMemset(d_in, 0, n*sizeof(float));
    int block = 256;
    int grid = (n + block*3 - 1) / (block*3);
    triple_buffer_kernel<<<grid, block>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    printf("triple buffer done\n");
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
