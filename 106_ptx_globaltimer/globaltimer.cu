#include <cuda_runtime.h>
#include <cstdio>

__device__ __forceinline__ unsigned long long globaltimer(){
    unsigned long long t;
    asm volatile("mov.u64 %0, %globaltimer;" : "=l"(t));
    return t;
}

__global__ void timer_kernel(unsigned long long* out){
    unsigned long long t0 = globaltimer();
    // small delay
    for (int i=0;i<1000;i++) asm volatile("add.u32 %0, %0, 1;" : "+r"(i));
    unsigned long long t1 = globaltimer();
    out[0] = t0;
    out[1] = t1;
}

int main(){
    unsigned long long* d = nullptr;
    unsigned long long h[2];
    cudaMalloc(&d, 2 * sizeof(unsigned long long));
    timer_kernel<<<1,1>>>(d);
    cudaMemcpy(h, d, 2 * sizeof(unsigned long long), cudaMemcpyDeviceToHost);
    printf("t0=%llu t1=%llu delta=%llu\n", h[0], h[1], h[1]-h[0]);
    cudaFree(d);
    return 0;
}
