#include <cstdio>
#include <vector>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// --- Utility ---------------------------------------------------------------
#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  std::exit(1);} } while(0)

static void init_host(std::vector<__half>& hA, std::vector<__half>& hB, int M, int N, int K)
{
    // Fill A(i,k) = (i+k) % 5, B(k,j) = (k+j) % 7 as halves
    for (int i=0;i<M;i++)
        for (int k=0;k<K;k++) {
            float v = float((i + k) % 5);
            hA[i*K + k] = __float2half(v);
        }
    for (int k=0;k<K;k++)
        for (int j=0;j<N;j++) {
            float v = float((k + j) % 7);
            hB[k*N + j] = __float2half(v);
        }
}

// --- Kernel 1: C++ WMMA API (16x16x16 tile per warp) ----------------------
//  A: row-major FP16, B: col-major FP16, C: row-major FP32 accumulator
//  compute C[MxN] = A[MxK] * B[KxN], M,N,K are multiples of 16 for demo.
__global__ void wmma_cxx_kernel(const __half* __restrict__ A,
                                const __half* __restrict__ B_colmajor,
                                float* __restrict__ C,
                                int M, int N, int K)
{
    // One warp -> one 16x16 tile of C
    int warp_m = blockIdx.y;   // tile row
    int warp_n = blockIdx.x;   // tile col

    // Fragments
    wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator,16,16,16, float> c_frag;

    wmma::fill_fragment(c_frag, 0.0f);

    // Base pointers for this tile in C
    int c_row = warp_m*16;
    int c_col = warp_n*16;

    // K loop in steps of 16
    for (int k0=0; k0<K; k0+=16) {
        const __half* tileA = A + (c_row * K + k0);              // row-major
        const __half* tileB = B_colmajor + (k0 * N + c_col);     // col-major view

        wmma::load_matrix_sync(a_frag, tileA, K);
        wmma::load_matrix_sync(b_frag, tileB, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store to C row-major
    wmma::store_matrix_sync(C + c_row*N + c_col, c_frag, N, wmma::mem_row_major);
}

// --- Kernel 2: tiny inline-PTX mma.sync exercise --------------------------
// This issues one mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
// with zeroed A/B operands (as a safe demo). It writes zeros into C,
// but SASS will include HMMA instructions to inspect the PTX/SASS path.
//
// NOTE: Doing a fully-correct GEMM purely in inline PTX requires careful
// per-thread register mapping & ldmatrix usage, which is advanced. This
// "stub" still rep inline-PTX calling convention safely.
//
// Launch with 1 warp (blockDim.x=32) on a 16x8 output region.
__global__ void wmma_inline_ptx_stub(float* __restrict__ C, int ldc)
{
    // 4 fp32 accum registers
    float acc[4] = {0.f, 0.f, 0.f, 0.f};

    // A: 4 regs (.f16x2 each), B: 2 regs (.f16x2 each) — zeroed for demo
    unsigned a[4] = {0,0,0,0};
    unsigned b[2] = {0,0};

    // Issue a single Tensor Core MMA:
    //   D = A*B + C, shape m16n8k16, A row-major, B col-major, acc in fp32
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, "
        "{%0,%1,%2,%3};\n"
        :
          "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        :
          "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1])
    );

    // Side-effect so compiler keeps the work
    int lane = threadIdx.x & 31;
    if (lane < 4) C[lane] = acc[lane];
}

// --- Host harness ----------------------------------------------------------
int main() {
    // small GEMM M=N=K=16 (single warp tile) for the WMMA 
    const int M=16, N=16, K=16;
    size_t bytesA = size_t(M)*K*sizeof(__half);
    size_t bytesB = size_t(K)*N*sizeof(__half);   // we’ll also keep a col-major copy
    size_t bytesC = size_t(M)*N*sizeof(float);

    std::vector<__half> hA(M*K), hB_row(K*N), hB_col(K*N);
    std::vector<float>  hC(M*N, 0.0f);

    init_host(hA, hB_row, M,N,K);

    // Build a col-major copy for B as WMMA expects (row.col variant)
    for (int k=0;k<K;k++)
      for (int j=0;j<N;j++)
          hB_col[k + j*K] = hB_row[k*N + j];

    __half *dA, *dBcol;
    float *dC;
    CHECK_CUDA(cudaMalloc(&dA, bytesA));
    CHECK_CUDA(cudaMalloc(&dBcol, bytesB));
    CHECK_CUDA(cudaMalloc(&dC, bytesC));
    CHECK_CUDA(cudaMemcpy(dA,    hA.data(),     bytesA, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dBcol, hB_col.data(), bytesB, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemset(dC, 0, bytesC));

    dim3 grid(N/16, M/16);   // 1x1 for 16x16
    dim3 block(32);          // one warp

    // Run C++ WMMA kernel (produces a real GEMM result)
    wmma_cxx_kernel<<<grid, block>>>(dA, dBcol, dC, M,N,K);
    CHECK_CUDA(cudaDeviceSynchronize());

    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    printf("[WMMA C++] C[0,0]=%f\n", hC[0]); // nonzero

    // Run the inline-PTX stub once (writes zeros into the first row region)
    wmma_inline_ptx_stub<<<1, 32>>>(dC, N);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(hC.data(), dC, bytesC, cudaMemcpyDeviceToHost));
    printf("[PTX stub] C[0,0]=%f (expect 0 after stub overwrite of first row)\n", hC[0]);

    cudaFree(dA); cudaFree(dBcol); cudaFree(dC);
    return 0;
}
