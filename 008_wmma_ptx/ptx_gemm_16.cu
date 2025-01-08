#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_fp16.h>

#define CHECK_CUDA(x) do { auto e=(x); if(e!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
  std::exit(1);} } while(0)

__device__ __forceinline__
void store_tile8(float* Cbase, int ldc, const float d[8]) {
  int lane = threadIdx.x & 31;
  int grp  = lane >> 2;      // 0..7
  int t    = lane & 3;       // 0..3
  int r0   = grp + ((t>>1) ? 8 : 0);
  int r1   = r0 + 4;
  int cols = (t & 1) ? 2 : 0;
  int rows[4] = {r0, r1, r0+2, r1+2};
  float* p;
  p = Cbase + rows[0]*ldc + cols; p[0]=d[0]; p[1]=d[1];
  p = Cbase + rows[1]*ldc + cols; p[0]=d[2]; p[1]=d[3];
  p = Cbase + rows[2]*ldc + cols; p[0]=d[4]; p[1]=d[5];
  p = Cbase + rows[3]*ldc + cols; p[0]=d[6]; p[1]=d[7];
}

// A, C row-major; B given as *row-major* KxN on host, but we’ll put two 16x8 halves in SMEM as row-major too
// then use ldmatrix.x2 (no .trans) to get the .col fragment B needs.
__global__ void gemm16x16x16_full_ptx(const __half* __restrict__ A,
                                      const __half* __restrict__ B_row,
                                      float* __restrict__ C,
                                      int lda, int ldb, int ldc)
{
  __shared__ __align__(16) __half As[16*16];     // row-major
  __shared__ __align__(16) __half Bs0[16*8];     // left half (cols 0..7), row-major in smem
  __shared__ __align__(16) __half Bs1[16*8];     // right half (cols 8..15), row-major in smem

  int lane = threadIdx.x & 31;

  // Load A (256 el): 8 per lane
  #pragma unroll
  for (int i=0;i<8;i++){
    int idx = lane + i*32;             // 0..255
    int r = idx / 16, c = idx % 16;
    As[idx] = A[r*lda + c];
  }

  // Load B halves from row-major KxN (K=16,N=16) into Bs0/Bs1 row-major 16x8
  // Left cols 0..7
  #pragma unroll
  for (int i=0;i<4;i++){
    int idx = lane + i*32;             // 0..127
    int r = idx % 16, c = idx / 16;    // r=0..15, c=0..7
    Bs0[r*8 + c] = B_row[r*ldb + c];
  }
  // Right cols 8..15
  #pragma unroll
  for (int i=0;i<4;i++){
    int idx = lane + i*32;
    int r = idx % 16, c = idx / 16;
    Bs1[r*8 + c] = B_row[r*ldb + (8+c)];
  }

  __syncthreads();

  // Per-lane SMEM addresses for ldmatrix
  // For A: As is row-major; to get A.row fragment → use .trans
  unsigned a_ptr = __cvta_generic_to_shared(&As[(lane%16)*16 + (lane/16)*8]);
  // For B: Bs* are row-major; to get B.col fragment → NO .trans
  unsigned b0_ptr = __cvta_generic_to_shared(&Bs0[(lane%16)*8 + (lane/16)*4]);
  unsigned b1_ptr = __cvta_generic_to_shared(&Bs1[(lane%16)*8 + (lane/16)*4]);

  // Registers
  unsigned a0,a1,a2,a3; // A: 4 regs
  unsigned b0,b1;       // B: 2 regs

  // A load: .x4.trans (row-major smem → .row fragment)
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
    : "r"(a_ptr)
  );

  // LEFT B load: .x2 (row-major smem → .col fragment)
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
    : "=r"(b0), "=r"(b1)
    : "r"(b0_ptr)
  );

  float dL[8];
  #pragma unroll
  for (int i=0;i<8;i++) dL[i]=0.0f;

  // MMA left 16x8 (m16n8k16, row.col): D += A(4 regs) * B(2 regs)
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7}, "
    "{%8,%9,%10,%11}, "
    "{%12,%13}, "
    "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
    :
      "+f"(dL[0]), "+f"(dL[1]), "+f"(dL[2]), "+f"(dL[3]),
      "+f"(dL[4]), "+f"(dL[5]), "+f"(dL[6]), "+f"(dL[7])
    :
      "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1)
  );

  // RIGHT B load
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
    : "=r"(b0), "=r"(b1)
    : "r"(b1_ptr)
  );

  float dR[8];
  #pragma unroll
  for (int i=0;i<8;i++) dR[i]=0.0f;

  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3,%4,%5,%6,%7}, "
    "{%8,%9,%10,%11}, "
    "{%12,%13}, "
    "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
    :
      "+f"(dR[0]), "+f"(dR[1]), "+f"(dR[2]), "+f"(dR[3]),
      "+f"(dR[4]), "+f"(dR[5]), "+f"(dR[6]), "+f"(dR[7])
    :
      "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1)
  );

  // Store back: left goes at col 0, right at col 8
  store_tile8(C + 0, ldc, dL);
  store_tile8(C + 8, ldc, dR);
}

// --------------------- host test ---------------------
int main(){
  constexpr int M=16,N=16,K=16;
  std::vector<__half> A(M*K), B_row(K*N);
  std::vector<float>  C(M*N,0.0f), Ref(M*N,0.0f);

  for(int i=0;i<M;i++) for(int k=0;k<K;k++) A[i*K+k]=__float2half((float)((i+k)%5));
  for(int k=0;k<K;k++) for(int j=0;j<N;j++) B_row[k*N+j]=__float2half((float)((k+j)%7));

  // CPU ref
  for(int i=0;i<M;i++)
    for(int j=0;j<N;j++){
      float s=0; for(int k=0;k<K;k++) s+=__half2float(A[i*K+k])*__half2float(B_row[k*N+j]);
      Ref[i*N+j]=s;
    }

  __half *dA,*dB; float *dC;
  CHECK_CUDA(cudaMalloc(&dA,M*K*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dB,K*N*sizeof(__half)));
  CHECK_CUDA(cudaMalloc(&dC,M*N*sizeof(float)));
  CHECK_CUDA(cudaMemcpy(dA,A.data(),M*K*sizeof(__half),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB,B_row.data(),K*N*sizeof(__half),cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(dC,0,M*N*sizeof(float)));

  gemm16x16x16_full_ptx<<<1,32>>>(dA,dB,dC,K,N,N);
  CHECK_CUDA(cudaDeviceSynchronize());
  CHECK_CUDA(cudaMemcpy(C.data(),dC,M*N*sizeof(float),cudaMemcpyDeviceToHost));

  printf("C[0,0]=%f ref=%f\n", C[0],Ref[0]);
  printf("C[0,1]=%f ref=%f\n", C[1],Ref[1]);
  printf("C[1,0]=%f ref=%f\n", C[N+0],Ref[N+0]);

  double maxe=0; for(int i=0;i<M*N;i++) maxe=fmax(maxe,fabs(C[i]-Ref[i]));
  printf("max |C-Ref| = %.3g\n", maxe);
  cudaFree(dA); cudaFree(dB); cudaFree(dC);
}
