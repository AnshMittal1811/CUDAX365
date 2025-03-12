#include <cuda_runtime.h>
#include <cstdio>

__device__ __forceinline__ int dp4a(int a, int b, int c){
    int out;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

__global__ void dp8a_kernel(const int* a, const int* b, int* out){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int acc = 0;
    acc = dp4a(a[i], b[i], acc);
    out[i] = acc;
}

int main(){
    int n = 256;
    int *da=nullptr, *db=nullptr, *dc=nullptr;
    cudaMalloc(&da, n*sizeof(int));
    cudaMalloc(&db, n*sizeof(int));
    cudaMalloc(&dc, n*sizeof(int));
    dp8a_kernel<<<1, 256>>>(da, db, dc);
    cudaDeviceSynchronize();
    printf("dp4a kernel done\n");
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    return 0;
}
