#include <cuda_runtime.h>
#include <cstdio>

__global__ void copy_pbo(const float* src, float* dst, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(v) : "l"(src + i));
    asm volatile("st.global.f32 [%0], %1;" :: "l"(dst + i), "f"(v));
}

int main(){
    int n = 1 << 20;
    float *d_src=nullptr, *d_dst=nullptr;
    cudaMalloc(&d_src, n*sizeof(float));
    cudaMalloc(&d_dst, n*sizeof(float));
    cudaMemset(d_src, 0, n*sizeof(float));
    copy_pbo<<<(n+255)/256, 256>>>(d_src, d_dst, n);
    cudaDeviceSynchronize();
    printf("copy done\n");
    cudaFree(d_src);
    cudaFree(d_dst);
    return 0;
}
