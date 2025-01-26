#include <cuda_runtime.h>
#include <mma.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cutensor.h>

using namespace nvcuda;

#define CHECK_CUDA(x) do { auto err=(x); if (err!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); return 1; } } while(0)
#define CHECK_CUTENSOR(x) do { auto st=(x); if (st!=CUTENSOR_STATUS_SUCCESS){ \
  printf("cuTensor error %s:%d: %s\n", __FILE__,__LINE__,cutensorGetErrorString(st)); return 1; } } while(0)

__global__ void wmma_gemm_kernel(const half* A, const half* B, float* C){
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
    const int M=16, N=16, K=16;
    size_t bytesA = M*K*sizeof(half);
    size_t bytesB = K*N*sizeof(half);
    size_t bytesC = M*N*sizeof(float);

    std::vector<half> hA(M*K), hB(K*N);
    for (int i=0;i<M*K;i++) hA[i] = __float2half((float)(i%7) * 0.1f);
    // store B in column-major for WMMA
    for (int r=0;r<K;r++) for (int c=0;c<N;c++) hB[c*K + r] = __float2half((float)((r+c)%5) * 0.2f);

    half *dA=nullptr, *dB=nullptr;
    float *dC=nullptr;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dB, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));
    CHECK_CUDA(cudaMemcpy(dA, hA.data(), bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    // WMMA
    wmma_gemm_kernel<<<1, 32>>>(dA, dB, dC);
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> hC_wmma(M*N, 0.0f);
    CHECK_CUDA(cudaMemcpy(hC_wmma.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    // cuTensor contraction
    cutensorHandle_t handle;
    CHECK_CUTENSOR(cutensorCreate(&handle));

    cutensorTensorDescriptor_t descA, descB, descC;
    int64_t extentA[2] = {M, K};
    int64_t extentB[2] = {K, N};
    int64_t extentC[2] = {M, N};
    int64_t strideA[2] = {K, 1};
    int64_t strideB[2] = {1, K}; // column-major
    int64_t strideC[2] = {N, 1};
    CHECK_CUTENSOR(cutensorCreateTensorDescriptor(&handle, &descA, 2, extentA, strideA, CUDA_R_16F, CUTENSOR_OP_IDENTITY));
    CHECK_CUTENSOR(cutensorCreateTensorDescriptor(&handle, &descB, 2, extentB, strideB, CUDA_R_16F, CUTENSOR_OP_IDENTITY));
    CHECK_CUTENSOR(cutensorCreateTensorDescriptor(&handle, &descC, 2, extentC, strideC, CUDA_R_32F, CUTENSOR_OP_IDENTITY));

    int32_t modeA[2] = {'m', 'k'};
    int32_t modeB[2] = {'k', 'n'};
    int32_t modeC[2] = {'m', 'n'};

    cutensorContractionDescriptor_t desc;
    CHECK_CUTENSOR(cutensorCreateContractionDescriptor(&handle, &desc,
        &descA, modeA, CUTENSOR_OP_IDENTITY,
        &descB, modeB, CUTENSOR_OP_IDENTITY,
        &descC, modeC, CUTENSOR_OP_IDENTITY,
        &descC, modeC, CUTENSOR_COMPUTE_32F));

    cutensorContractionFind_t find;
    CHECK_CUTENSOR(cutensorInitContractionFind(&handle, &find, CUTENSOR_ALGO_DEFAULT));

    size_t worksize = 0;
    CHECK_CUTENSOR(cutensorContractionGetWorkspaceSize(&handle, &desc, &find, CUTENSOR_WORKSPACE_RECOMMENDED, &worksize));
    void* workspace = nullptr;
    if (worksize) CHECK_CUDA(cudaMalloc(&workspace, worksize));

    cutensorContractionPlan_t plan;
    CHECK_CUTENSOR(cutensorInitContractionPlan(&handle, &plan, &desc, &find, worksize));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUTENSOR(cutensorContraction(&handle, &plan,
        &alpha, dA, dB, &beta, dC, dC, workspace, worksize, 0));
    CHECK_CUDA(cudaDeviceSynchronize());

    std::vector<float> hC_cutensor(M*N, 0.0f);
    CHECK_CUDA(cudaMemcpy(hC_cutensor.data(), dC, bytesC, cudaMemcpyDeviceToHost));

    // compare
    double err = 0.0;
    for (int i=0;i<M*N;i++){
        double diff = hC_cutensor[i] - hC_wmma[i];
        err += diff * diff;
    }
    printf("L2 error = %.6e\n", std::sqrt(err));

    if (workspace) cudaFree(workspace);
    cutensorDestroy(handle);
    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
