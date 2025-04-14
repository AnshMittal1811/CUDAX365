#include <cuda_runtime.h>
#include <cstdio>

__global__ void pde_ref(const float* in, float* out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = in[i] * 0.99f;
}


int main(){
    int n = 1 << 16;
    float* d_in=nullptr; float* d_out=nullptr;
    cudaMalloc(&d_in, n*sizeof(float));
    cudaMalloc(&d_out, n*sizeof(float));
    pde_ref<<<(n+255)/256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    printf("done\n");
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
