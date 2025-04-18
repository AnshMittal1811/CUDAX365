#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cuda_runtime.h>
#include <cstdio>



int main(){
    using ElementInput = cutlass::half_t;
    using ElementOutput = float;
    using Layout = cutlass::layout::RowMajor;

    using Gemm = cutlass::gemm::device::Gemm<
        ElementInput, Layout,
        ElementInput, Layout,
        ElementOutput, Layout
    >;

    int m = 128, n = 128, k = 128;
    size_t bytesA = m * k * sizeof(ElementInput);
    size_t bytesB = k * n * sizeof(ElementInput);
    size_t bytesC = m * n * sizeof(ElementOutput);

    ElementInput *dA=nullptr, *dB=nullptr;
    ElementOutput *dC=nullptr;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);
    cudaMemset(dA, 0, bytesA);
    cudaMemset(dB, 0, bytesB);
    cudaMemset(dC, 0, bytesC);

    Gemm gemm_op;
    Gemm::Arguments args({m, n, k}, {dA, k}, {dB, n}, {dC, n}, {dC, n});
    cutlass::Status st = gemm_op(args);
    if (st != cutlass::Status::kSuccess){
        printf("CUTLASS GEMM failed\n");
        return 1;
    }
    cudaDeviceSynchronize();
    printf("GEMM done\n");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
