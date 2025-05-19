#include <cutensor.h>
#include <cuda_runtime.h>
#include <cstdio>

#define CHECK_CUDA(x) do { auto err=(x); if (err!=cudaSuccess){ \
  printf("CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); return 1; } } while(0)
#define CHECK_CUTENSOR(x) do { auto st=(x); if (st!=CUTENSOR_STATUS_SUCCESS){ \
  printf("cuTensor error %s:%d: %s\n", __FILE__,__LINE__,cutensorGetErrorString(st)); return 1; } } while(0)

int main(){
    int64_t extentA[4] = {16, 16, 4, 4};
    int64_t extentB[4] = {16, 16, 4, 4};
    int64_t extentC[4] = {16, 16, 4, 4};
    int64_t strideA[4] = {16*4*4, 4*4, 4, 1};
    int64_t strideB[4] = {16*4*4, 4*4, 4, 1};
    int64_t strideC[4] = {16*4*4, 4*4, 4, 1};

    cutensorHandle_t handle;
    CHECK_CUTENSOR(cutensorCreate(&handle));

    cutensorTensorDescriptor_t descA, descB, descC;
    CHECK_CUTENSOR(cutensorCreateTensorDescriptor(&handle, &descA, 4, extentA, strideA, CUDA_R_32F, CUTENSOR_OP_IDENTITY));
    CHECK_CUTENSOR(cutensorCreateTensorDescriptor(&handle, &descB, 4, extentB, strideB, CUDA_R_32F, CUTENSOR_OP_IDENTITY));
    CHECK_CUTENSOR(cutensorCreateTensorDescriptor(&handle, &descC, 4, extentC, strideC, CUDA_R_32F, CUTENSOR_OP_IDENTITY));

    int32_t modeA[4] = {'a','b','c','d'};
    int32_t modeB[4] = {'a','b','c','d'};
    int32_t modeC[4] = {'a','b','c','d'};

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

    size_t bytes = 16*16*4*4*sizeof(float);
    float *A=nullptr, *B=nullptr, *C=nullptr;
    CHECK_CUDA(cudaMalloc(&A, bytes));
    CHECK_CUDA(cudaMalloc(&B, bytes));
    CHECK_CUDA(cudaMalloc(&C, bytes));

    float alpha = 1.0f, beta = 0.0f;
    CHECK_CUTENSOR(cutensorContraction(&handle, &plan, &alpha, A, B, &beta, C, C, workspace, worksize, 0));
    CHECK_CUDA(cudaDeviceSynchronize());
    printf("cutensor 4D contraction done\n");

    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    if (workspace) cudaFree(workspace);
    cutensorDestroy(handle);
    return 0;
}
