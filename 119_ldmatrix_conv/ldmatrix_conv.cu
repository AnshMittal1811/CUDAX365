#include <mma.h>
#include <cuda_runtime.h>
#include <cstdio>

using namespace nvcuda;

__global__ void wmma_matmul(const half* A, const half* B, float* C){
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.0f);
    wmma::load_matrix_sync(a_frag, A, 16);
    wmma::load_matrix_sync(b_frag, B, 16);
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    wmma::store_matrix_sync(C, c_frag, 16, wmma::mem_row_major);
}

int main(){
    half *A=nullptr, *B=nullptr;
    float *C=nullptr;
    cudaMalloc(&A, 16*16*sizeof(half));
    cudaMalloc(&B, 16*16*sizeof(half));
    cudaMalloc(&C, 16*16*sizeof(float));
    wmma_matmul<<<1, 32>>>(A, B, C);
    cudaDeviceSynchronize();
    printf("wmma done\n");
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
