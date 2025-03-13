#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdio>

#define CHECK_CUDA(x) do { auto err=(x); if (err!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); return 1; } } while(0)
#define CHECK_CUBLAS(x) do { auto st=(x); if (st!=CUBLAS_STATUS_SUCCESS){ \
  printf("cuBLASLt error %s:%d\n", __FILE__,__LINE__); return 1; } } while(0)

int main(){
    const int m=256, n=256, k=256;
    size_t bytes = m*k*sizeof(__half);
    __half *A=nullptr, *B=nullptr;
    float *C=nullptr;
    CHECK_CUDA(cudaMalloc(&A, bytes));
    CHECK_CUDA(cudaMalloc(&B, bytes));
    CHECK_CUDA(cudaMalloc(&C, m*n*sizeof(float)));

    cublasLtHandle_t handle;
    CHECK_CUBLAS(cublasLtCreate(&handle));

    cublasOperation_t opT = CUBLAS_OP_T;
    cublasOperation_t opN = CUBLAS_OP_N;

    cudaDataType_t a_type = CUDA_R_16F;
    cudaDataType_t b_type = CUDA_R_16F;
    cudaDataType_t c_type = CUDA_R_32F;
    cublasComputeType_t compute = CUBLAS_COMPUTE_32F;

#if defined(CUDA_R_8F_E4M3)
    a_type = CUDA_R_8F_E4M3;
    b_type = CUDA_R_8F_E4M3;
    compute = CUBLAS_COMPUTE_32F_FAST_TF32;
#endif

    cublasLtMatmulDesc_t matmul_desc;
    CHECK_CUBLAS(cublasLtMatmulDescCreate(&matmul_desc, compute, c_type));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &opN, sizeof(opN)));
    CHECK_CUBLAS(cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof(opN)));

    cublasLtMatrixLayout_t a_layout, b_layout, c_layout;
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&a_layout, a_type, m, k, m));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&b_layout, b_type, k, n, k));
    CHECK_CUBLAS(cublasLtMatrixLayoutCreate(&c_layout, c_type, m, n, m));

    float alpha = 1.0f, beta = 0.0f;

    cublasLtMatmulPreference_t pref;
    CHECK_CUBLAS(cublasLtMatmulPreferenceCreate(&pref));
    size_t workspace_size = 1 << 20;
    CHECK_CUBLAS(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)));

    cublasLtMatmulHeuristicResult_t heuristic = {};
    int returned = 0;
    CHECK_CUBLAS(cublasLtMatmulAlgoGetHeuristic(handle, matmul_desc, a_layout, b_layout, c_layout, c_layout, pref, 1, &heuristic, &returned));
    if (returned == 0){
        printf("no heuristic found\n");
        return 1;
    }

    void* workspace = nullptr;
    CHECK_CUDA(cudaMalloc(&workspace, workspace_size));

    CHECK_CUBLAS(cublasLtMatmul(handle, matmul_desc, &alpha, A, a_layout, B, b_layout, &beta, C, c_layout, C, c_layout, &heuristic.algo, workspace, workspace_size, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("cublasLt matmul done\n");

    cudaFree(workspace);
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(a_layout);
    cublasLtMatrixLayoutDestroy(b_layout);
    cublasLtMatrixLayoutDestroy(c_layout);
    cublasLtMatmulDescDestroy(matmul_desc);
    cublasLtDestroy(handle);

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    return 0;
}
