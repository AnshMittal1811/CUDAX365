// We *declare* the kernel here; device body lives in PTX we'll link in.
extern "C" __global__
void vec_add_kernel(const float*, const float*, float*, int);

#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do{auto err=(x); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); \
  exit(1);} }while(0)

float run_once(int N, float *dA, float *dB, float *dC) {
  dim3 block(256);
  dim3 grid((N + block.x - 1)/block.x);

  // warmup
  for(int i=0;i<5;i++) vec_add_kernel<<<grid,block>>>(dA,dB,dC,N);
  CHECK_CUDA(cudaDeviceSynchronize());

  cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
  CHECK_CUDA(cudaEventRecord(s));
  for(int it=0; it<200; ++it) vec_add_kernel<<<grid,block>>>(dA,dB,dC,N);
  CHECK_CUDA(cudaEventRecord(e));
  CHECK_CUDA(cudaEventSynchronize(e));
  float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,s,e));
  CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
  return ms;
}

int main(){
  const int N = 1<<26; // ~67M elements
  const size_t bytes = N * sizeof(float);

  float *hA=(float*)malloc(bytes), *hB=(float*)malloc(bytes);
  for (int i=0;i<N;i++){ hA[i]=1.0f; hB[i]=2.0f; }

  float *dA,*dB,*dC;
  CHECK_CUDA(cudaMalloc(&dA,bytes));
  CHECK_CUDA(cudaMalloc(&dB,bytes));
  CHECK_CUDA(cudaMalloc(&dC,bytes));
  CHECK_CUDA(cudaMemcpy(dA,hA,bytes,cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dB,hB,bytes,cudaMemcpyHostToDevice));

  float ms = run_once(N,dA,dB,dC);

  // check result
  float out0; CHECK_CUDA(cudaMemcpy(&out0,dC,sizeof(float),cudaMemcpyDeviceToHost));
  printf("C[0]=%.1f (expect 3.0)\n", out0);

  // throughput lower bound (2 loads + 1 store per element) * 200 iters
  double bytes_moved = (double)N * 3 * sizeof(float) * 200;
  double gbps = bytes_moved / (ms/1000.0) / 1e9;
  printf("Time %.3f ms for 200 iters  ~ %.2f GB/s\n", ms, gbps);

  cudaFree(dA); cudaFree(dB); cudaFree(dC); free(hA); free(hB);
  return 0;
}
