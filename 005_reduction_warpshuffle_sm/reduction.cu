#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  exit(1);} } while(0)

constexpr int WARP_SIZE = 32;

__inline__ __device__ float warp_reduce_sum(float v) {
    // intra-warp reduction using shfl_down
    unsigned mask = 0xffffffffu;
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(mask, v, offset);
    }
    return v;
}

// ------------------- Kernel 1: Shared-memory tree reduction -------------------
template <unsigned BLOCK_SIZE>
__global__ void reduce_shared(const float* __restrict__ x, size_t N, float* __restrict__ out)
{
    __shared__ float sdata[BLOCK_SIZE];

    size_t tid  = threadIdx.x;
    size_t idx  = blockIdx.x * BLOCK_SIZE * 2 + tid; // load 2 elements per thread
    float sum = 0.0f;

    if (idx < N) sum += x[idx];
    if (idx + BLOCK_SIZE < N) sum += x[idx + BLOCK_SIZE];

    sdata[tid] = sum;
    __syncthreads();

    // classic tree reduction in shared memory
    if (BLOCK_SIZE >= 1024) { if (tid < 512) sdata[tid] += sdata[tid + 512]; __syncthreads(); }
    if (BLOCK_SIZE >=  512) { if (tid < 256) sdata[tid] += sdata[tid + 256]; __syncthreads(); }
    if (BLOCK_SIZE >=  256) { if (tid < 128) sdata[tid] += sdata[tid + 128]; __syncthreads(); }
    if (BLOCK_SIZE >=  128) { if (tid <  64) sdata[tid] += sdata[tid +  64]; __syncthreads(); }

    // last 64 threads (2 warps) – unrolled without __syncthreads (same warp)
    if (tid < 32) {
        volatile float* vsmem = sdata; // allow compiler to keep in smem
        if (BLOCK_SIZE >=  64) vsmem[tid] += vsmem[tid + 32];
        if (BLOCK_SIZE >=  32) vsmem[tid] += vsmem[tid + 16];
        if (BLOCK_SIZE >=  16) vsmem[tid] += vsmem[tid +  8];
        if (BLOCK_SIZE >=   8) vsmem[tid] += vsmem[tid +  4];
        if (BLOCK_SIZE >=   4) vsmem[tid] += vsmem[tid +  2];
        if (BLOCK_SIZE >=   2) vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) atomicAdd(out, sdata[0]);
}

// ------------------- Kernel 2: Warp-shuffle reduction -------------------
template <unsigned BLOCK_SIZE>
__global__ void reduce_shuffle_kernel(const float* __restrict__ x, size_t N, float* __restrict__ out)
{
    float sum = 0.0f;
    size_t idx  = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
    if (idx < N) sum += x[idx];
    if (idx + BLOCK_SIZE < N) sum += x[idx + BLOCK_SIZE];

    // intra-warp reduce
    sum = warp_reduce_sum(sum);

    // one value per warp -> shared
    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0) {
        warp_sums[threadIdx.x / WARP_SIZE] = sum;
    }
    __syncthreads();

    // final reduce by warp 0
    float block_sum = 0.0f;
    if (threadIdx.x < BLOCK_SIZE / WARP_SIZE) {
        block_sum = warp_sums[threadIdx.x];
    }
    if (threadIdx.x < WARP_SIZE) {
        block_sum = warp_reduce_sum(block_sum);
    }

    if (threadIdx.x == 0) atomicAdd(out, block_sum);
}

// ------------------- Timing harness -------------------
template <typename Kernel>
float run_and_time(const char* name, Kernel k, const float* dX, size_t N, float* dOut,
                   int iters, int block_size)
{
    CHECK_CUDA(cudaMemset(dOut, 0, sizeof(float)));

    dim3 block(block_size);
    dim3 grid((int)((N + block.x * 2 - 1) / (block.x * 2)));

    // Warmup
    for (int i=0;i<5;i++) k<<<grid, block>>>(dX, N, dOut);
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t start, stop; CHECK_CUDA(cudaEventCreate(&start)); CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    for (int i=0;i<iters;i++) k<<<grid, block>>>(dX, N, dOut);
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    float ms=0.0f; CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start)); CHECK_CUDA(cudaEventDestroy(stop));

    // Approx memory traffic (read 2 elements per thread; one atomic per block)
    double bytes = (double)N * sizeof(float) * iters; // lower bound
    double gbps = bytes / (ms/1000.0) / 1e9;
    printf("%s: %7.3f ms for %d iters, ~%.2f GB/s\n", name, ms, iters, gbps);
    return ms;
}

int main() {
    const size_t N = (size_t)1 << 26; // ~67M
    const int    BLOCK = 256;
    const int    ITERS = 200;

    // Host init
    float* hX = (float*)malloc(N * sizeof(float));
    for (size_t i=0;i<N;i++) hX[i] = 1.0f;

    // Device
    float *dX, *dOut;
    CHECK_CUDA(cudaMalloc(&dX, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dX, hX, N * sizeof(float), cudaMemcpyHostToDevice));

    // Launch typed kernels through lambdas for the timing wrapper
    auto K_shared  = [] __device__ (const float*, size_t, float*) {};
    auto K_shuffle = [] __device__ (const float*, size_t, float*) {};
    (void)K_shared; (void)K_shuffle; // silence unused

    // Use explicit template instantiation to choose BLOCK size
    float ms1 = run_and_time("reduce_shared",
        reduce_shared<BLOCK>, dX, N, dOut, ITERS, BLOCK);

    float ms2 = run_and_time("reduce_shuffle",
        reduce_shuffle_kernel<BLOCK>, dX, N, dOut, ITERS, BLOCK);

    // Correctness check
    float sum=0.0f;
    CHECK_CUDA(cudaMemcpy(&sum, dOut, sizeof(float), cudaMemcpyDeviceToHost));
    printf("Check: sum=%.1f (expected %.1f)\n", sum, (double)N);

    cudaFree(dX); cudaFree(dOut);
    free(hX);
    return 0;
}
