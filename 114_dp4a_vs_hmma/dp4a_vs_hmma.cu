#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>

using namespace nvcuda;

__device__ __forceinline__ int dp4a(int a, int b, int c){
    int out;
    asm volatile("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(out) : "r"(a), "r"(b), "r"(c));
    return out;
}

__global__ void dp4a_kernel(const int* a, const int* b, int* out){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int acc = 0;
    acc = dp4a(a[i], b[i], acc);
    out[i] = acc;
}

__global__ void wmma_kernel(const half* A, const half* B, float* C){
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

float time_kernel_dp4a(){
    int n = 256;
    int *a=nullptr, *b=nullptr, *c=nullptr;
    cudaMalloc(&a, n*sizeof(int));
    cudaMalloc(&b, n*sizeof(int));
    cudaMalloc(&c, n*sizeof(int));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    dp4a_kernel<<<1, 256>>>(a, b, c);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaFree(a); cudaFree(b); cudaFree(c);
    return ms;
}

float time_kernel_wmma(){
    half *A=nullptr, *B=nullptr;
    float *C=nullptr;
    cudaMalloc(&A, 16*16*sizeof(half));
    cudaMalloc(&B, 16*16*sizeof(half));
    cudaMalloc(&C, 16*16*sizeof(float));
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    wmma_kernel<<<1, 32>>>(A, B, C);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    cudaFree(A); cudaFree(B); cudaFree(C);
    return ms;
}

int main(){
    float t_dp4a = time_kernel_dp4a();
    float t_wmma = time_kernel_wmma();
    printf("dp4a_ms=%.6f wmma_ms=%.6f\n", t_dp4a, t_wmma);
    return 0;
}
