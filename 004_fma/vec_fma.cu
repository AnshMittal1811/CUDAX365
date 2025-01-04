#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  exit(1);} } while(0)

__global__ void add_baseline(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) C[i] = A[i] + B[i];
}

__global__ void add_inline_ptx_fma(const float* __restrict__ A,
                                   const float* __restrict__ B,
                                   float* __restrict__ C,
                                   int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        float a = A[i], b = B[i], out;
        asm volatile ("fma.rn.f32 %0, %1, %2, %3;\n"
                      : "=f"(out) : "f"(b), "f"(1.0f), "f"(a));
        C[i] = out;
    }
}

// Make the kernel a non-type template parameter (function pointer)
template <void (*K)(const float*, const float*, float*, int)>
void run_and_time(const char* name,
                  const float* dA, const float* dB, float* dC, int N)
{
    dim3 block(256);
    dim3 grid((N + block.x - 1)/block.x);

    for (int i=0;i<5;i++) K<<<grid, block>>>(dA, dB, dC, N);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));

    CHECK_CUDA(cudaEventRecord(start));
    for (int it=0; it<100; ++it) K<<<grid, block>>>(dA, dB, dC, N);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms=0.0f; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    double bytes_per_iter = 3.0 * sizeof(float) * (double)N;
    double gbps = (bytes_per_iter * 100) / (ms/1000.0) / 1e9;
    printf("%s: %f ms for 100 iters  (~%.2f GB/s)\n", name, ms, gbps);

    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
}

int main(){
    const int N = 1<<26; // ~67M
    size_t bytes = (size_t)N * sizeof(float);

    float *hA = (float*)malloc(bytes);
    float *hB = (float*)malloc(bytes);
    for (int i=0;i<N;i++){ hA[i] = 1.0f; hB[i] = 2.0f; }

    float *dA,*dB,*dC;
    CHECK_CUDA(cudaMalloc(&dA, bytes));
    CHECK_CUDA(cudaMalloc(&dB, bytes));
    CHECK_CUDA(cudaMalloc(&dC, bytes));
    CHECK_CUDA(cudaMemcpy(dA, hA, bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(dB, hB, bytes, cudaMemcpyHostToDevice));

    run_and_time<add_baseline>("baseline_add", dA, dB, dC, N);
    run_and_time<add_inline_ptx_fma>("inline_ptx_fma", dA, dB, dC, N);

    float hc=0.0f; CHECK_CUDA(cudaMemcpy(&hc, dC, sizeof(float), cudaMemcpyDeviceToHost));
    printf("C[0]=%f (expect 3.0)\n", hc);

    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    free(hA); free(hB);
    return 0;
}
