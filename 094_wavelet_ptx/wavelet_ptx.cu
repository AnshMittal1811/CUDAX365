#include <cuda_runtime.h>
#include <cstdio>

__device__ __forceinline__ float add_rn(float a, float b){
    float out;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__global__ void haar_step(const float* in, float* out, int n){
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 >= n) return;
    float a = in[i];
    float b = in[i+1];
    float avg = 0.5f * add_rn(a, b);
    float diff = 0.5f * add_rn(a, -b);
    out[i/2] = avg;
    out[n/2 + i/2] = diff;
}

int main(){
    int n = 1024;
    float* d_in=nullptr; float* d_out=nullptr;
    cudaMalloc(&d_in, n*sizeof(float));
    cudaMalloc(&d_out, n*sizeof(float));
    cudaMemset(d_in, 0, n*sizeof(float));
    haar_step<<<(n/2+255)/256, 256>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    printf("wavelet step done\n");
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
