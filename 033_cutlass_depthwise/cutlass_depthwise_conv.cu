#include <cutlass/cutlass.h>
#include <cutlass/conv/device/implicit_gemm_convolution.h>
#include <cutlass/layout/tensor.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

int main(){
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;
    using LayoutInputA = cutlass::layout::TensorNHWC;
    using LayoutInputB = cutlass::layout::TensorNHWC;
    using LayoutOutput = cutlass::layout::TensorNHWC;

    using Conv2d = cutlass::conv::device::ImplicitGemmConvolution<
        ElementInputA, LayoutInputA,
        ElementInputB, LayoutInputB,
        ElementOutput, LayoutOutput,
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<64, 64, 32>,
        cutlass::gemm::GemmShape<32, 32, 32>,
        cutlass::gemm::GemmShape<16, 8, 16>,
        cutlass::epilogue::thread::LinearCombination<
            ElementOutput, 8, ElementAccumulator, ElementAccumulator>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        2,
        cutlass::conv::Operator::kFprop
    >;

    int N=1, H=8, W=8, C=8;  // depthwise: K=C
    int K=C, R=3, S=3;
    int P=H, Q=W;
    int pad=1, stride=1, dilation=1;
    int groups = C;

    cutlass::conv::Conv2dProblemSize problem_size(
        {N, H, W, C}, {K, R, S, C}, {pad, pad}, {stride, stride}, {dilation, dilation},
        {N, P, Q, K}, cutlass::conv::Mode::kCrossCorrelation, groups);

    size_t bytesA = N*H*W*C*sizeof(ElementInputA);
    size_t bytesB = K*R*S*C*sizeof(ElementInputB);
    size_t bytesC = N*P*Q*K*sizeof(ElementOutput);

    ElementInputA *dA=nullptr;
    ElementInputB *dB=nullptr;
    ElementOutput *dC=nullptr;
    cudaMalloc(&dA, bytesA);
    cudaMalloc(&dB, bytesB);
    cudaMalloc(&dC, bytesC);
    cudaMemset(dA, 0, bytesA);
    cudaMemset(dB, 0, bytesB);
    cudaMemset(dC, 0, bytesC);

    typename Conv2d::Arguments args(
        problem_size,
        {dA, LayoutInputA::Stride::packed({H, W, C})},
        {dB, LayoutInputB::Stride::packed({R, S, C})},
        {dC, LayoutOutput::Stride::packed({P, Q, K})},
        {dC, LayoutOutput::Stride::packed({P, Q, K})},
        {1.0f, 0.0f}
    );

    Conv2d conv_op;
    cutlass::Status status = conv_op.initialize(args);
    if (status != cutlass::Status::kSuccess){
        printf("CUTLASS init failed\n");
        return 1;
    }
    status = conv_op();
    if (status != cutlass::Status::kSuccess){
        printf("CUTLASS run failed\n");
        return 1;
    }
    cudaDeviceSynchronize();
    printf("depthwise conv done\n");

    cudaFree(dA);
    cudaFree(dB);
    cudaFree(dC);
    return 0;
}
