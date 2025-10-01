#include <cstdio>
#include <cstdlib>          // atoi
#include <vector>
#include <cuda_fp16.h>
#include <mma.h>
#include <cublas_v2.h>
using namespace nvcuda;

#define CHECK_CUDA(x) do { cudaError_t err=(x); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)
#define CHECK_CUBLAS(x) do { cublasStatus_t st=(x); if(st!=CUBLAS_STATUS_SUCCESS){ \
  fprintf(stderr,"cuBLAS error %s:%d: %d\n", __FILE__,__LINE__,(int)st); exit(1);} } while(0)

// ----- WMMA 16x16x16 (A row-major FP16, B col-major FP16, C row-major FP32) ---
__global__ void wmma_gemm_kernel(const __half* __restrict__ Arow,
                                 const __half* __restrict__ Bcol,
                                 float* __restrict__ Crow,
                                 int M, int N, int K)
{
    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;

    wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> b;
    wmma::fragment<wmma::accumulator,16,16,16, float> c;
    wmma::fill_fragment(c, 0.0f);

    int baseRow = tile_m*16;
    int baseCol = tile_n*16;

    for (int k0=0; k0<K; k0+=16) {
        const __half* Ap = Arow + baseRow*K + k0;   // row-major
        const __half* Bp = Bcol + k0*N + baseCol;   // col-major
        wmma::load_matrix_sync(a, Ap, K);
        wmma::load_matrix_sync(b, Bp, N);
        wmma::mma_sync(c, a, b, c);
    }
    wmma::store_matrix_sync(Crow + baseRow*N + baseCol, c, N, wmma::mem_row_major);
}

// ----- Helpers --------------------------------------------------------------
static void fill_row_major(std::vector<__half>& Arow, int M, int K){
    for (int i=0;i<M;i++)
      for (int k=0;k<K;k++)
        Arow[i*K + k] = __float2half( (float)((i + 2*k) % 7) * 0.5f );
}
static void fill_row_major_B(std::vector<__half>& Brow, int K, int N){
    for (int k=0;k<K;k++)
      for (int j=0;j<N;j++)
        Brow[k*N + j] = __float2half( (float)((3*k + j) % 5) * 0.25f );
}
static void row_to_col(const std::vector<__half>& Row, std::vector<__half>& Col, int rows, int cols){
    for (int r=0;r<rows;r++)
      for (int c=0;c<cols;c++)
        Col[r + c*rows] = Row[r*cols + c];
}

// Generic timer – works with lambdas/functors, no <functional> needed
template <typename F>
float time_ms(F&& f, int iters=100, int warmup=10){
    cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
    for(int i=0;i<warmup;i++) f();
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(s));
    for(int i=0;i<iters;i++) f();
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float ms=0.0f; CHECK_CUDA(cudaEventElapsedTime(&ms,s,e));
    CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
    return ms;
}

int main(int argc, char** argv)
{
    int M = (argc>1)? atoi(argv[1]) : 128;
    int N = (argc>2)? atoi(argv[2]) : 128;
    int K = (argc>3)? atoi(argv[3]) : 128;
    if (M%16 || N%16 || K%16) { printf("M,N,K must be multiples of 16.\n"); return 0; }

    // Host buffers
    std::vector<__half> hA_row(M*K), hB_row(K*N);
    std::vector<__half> hA_col(M*K), hB_col(K*N);
    std::vector<float>  hC_row(M*N, 0.0f), hC_col(M*N, 0.0f);
    fill_row_major(hA_row, M, K);
    fill_row_major_B(hB_row, K, N);
    row_to_col(hA_row, hA_col, M, K);
    row_to_col(hB_row, hB_col, K, N);

    // Device buffers
    __half *dA_row, *dB_col, *dA_col, *dB_col_cm;
    float *dC_row, *dC_col;
    CHECK_CUDA(cudaMalloc(&dA_row,   (size_t)M*K*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_col,   (size_t)K*N*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_row,   (size_t)M*N*sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dA_col,   (size_t)M*K*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dB_col_cm,(size_t)K*N*sizeof(__half)));
    CHECK_CUDA(cudaMalloc(&dC_col,   (size_t)M*N*sizeof(float)));

    CHECK_CUDA(cudaMemcpy(dA_row, hA_row.data(), (size_t)M*K*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_col, hB_col.data(), (size_t)K*N*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dA_col, hA_col.data(), (size_t)M*K*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB_col_cm, hB_col.data(), (size_t)K*N*sizeof(__half), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC_row, 0, (size_t)M*N*sizeof(float)));
    CHECK_CUDA(cudaMemset(dC_col, 0, (size_t)M*N*sizeof(float)));

    // WMMA timing
    dim3 block(32);                // 1 warp
    dim3 grid(N/16, M/16);         // one 16x16 tile per warp
    auto wmma_run = [&](){ wmma_gemm_kernel<<<grid, block>>>(dA_row, dB_col, dC_row, M,N,K); };
    float ms_wmma = time_ms(wmma_run, 200, 20);
    double flops = 2.0 * (double)M * (double)N * (double)K;
    double tflops_wmma = (flops * 200.0) / (ms_wmma/1000.0) / 1e12;
    printf("[WMMA] %dx%dx%d: %f ms (200 iters)  ~ %.3f TFLOP/s\n", M,N,K, ms_wmma, tflops_wmma);

    // cuBLAS timing
    cublasHandle_t h; CHECK_CUBLAS(cublasCreate(&h));
    CHECK_CUBLAS(cublasSetMathMode(h, CUBLAS_TENSOR_OP_MATH));
    const float alpha=1.0f, beta=0.0f;
    auto cublas_run = [&](){
        CHECK_CUBLAS(cublasGemmEx(
            h, CUBLAS_OP_N, CUBLAS_OP_N,
            M, N, K,
            &alpha,
            dA_col,    CUDA_R_16F, M,   // col-major A
            dB_col_cm, CUDA_R_16F, K,   // col-major B
            &beta,
            dC_col,    CUDA_R_32F, M,   // col-major C (float accum)
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    };
    float ms_blas = time_ms(cublas_run, 200, 20);
    double tflops_blas = (flops * 200.0) / (ms_blas/1000.0) / 1e12;
    printf("[cuBLAS] %dx%dx%d: %f ms (200 iters)  ~ %.3f TFLOP/s\n", M,N,K, ms_blas, tflops_blas);

    // Sample check
    CHECK_CUDA(cudaMemcpy(hC_row.data(), dC_row, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(hC_col.data(), dC_col, (size_t)M*N*sizeof(float), cudaMemcpyDeviceToHost));
    printf("Samples  Cwmma[0]=%.3f   Ccublas[0]=%.3f\n", hC_row[0], hC_col[0]);

    cublasDestroy(h);
    cudaFree(dA_row); cudaFree(dB_col); cudaFree(dC_row);
    cudaFree(dA_col); cudaFree(dB_col_cm); cudaFree(dC_col);
    return 0;
}
