#include <cstudio.h>
#include <cuda.h>
#include <cuda_fp16.h>
#include <vector>

#define CHECK_CUDA(x) do { auto err = (x); if (err != cudaSuccess) { \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  std::exit(1);} } while(0)

// A, C: row-major (M=N=K=16) ; B: col-major (K×N)
__global__ void gemm16x16x16_full_ptx(const __half* __restrict__ A,
                                      const __half* __restrict__ B_col,
                                      float* __restrict__ C,
                                      int lda, int ldb, int ldc)
    {
        // one warp handles a 16x16 tile at (row=0,col=0)
        __shared__ __align__(16) __half As[16*16];
        __shared__ __align__(16) __half Bs0[16*8];  // left half of B tile (cols 0..7)
        __shared__ __align__(16) __half Bs1[16*8];  // right half (cols 8..15)

        const int lane = threadIdx.x & 31;


        // Load A (16 x 16 row-major) to shared memory (coalesced by rows, 16x16 = 256 el)
        // Each lane loads 8 elements (2 rows of 4 el = 32 * 8 = 256)
        for (int i=0; i<8; i++) {
            int idx = lane + i*32; // 0..255
            int r = idx / 16;      // row 0..15
            int c = idx % 16;      // col 0..15
            As[r*16 + c] = A[r*lda + c];
        }

        // Load B (left half) (cols 0..7) to col-major into shared (16x8)

    }