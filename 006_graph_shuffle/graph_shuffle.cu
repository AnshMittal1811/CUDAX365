#include <cstdio>
#include <cuda_runtime.h>

#define CHECK_CUDA(x) do { cudaError_t err = (x); if (err != cudaSuccess) { \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
  exit(1);} } while(0)

constexpr int WARP_SIZE = 32;

__inline__ __device__ float warp_reduce_sum(float v) {
    unsigned mask = 0xffffffffu;
    for (int offset = WARP_SIZE >> 1; offset > 0; offset >>= 1)
        v += __shfl_down_sync(mask, v, offset);
    return v;
}

template <unsigned BLOCK_SIZE>
__global__ void reduce_shuffle_kernel(const float* __restrict__ x, size_t N, float* __restrict__ out)
{
    float sum = 0.0f;
    size_t idx  = blockIdx.x * BLOCK_SIZE * 2 + threadIdx.x;
    if (idx < N) sum += x[idx];
    if (idx + BLOCK_SIZE < N) sum += x[idx + BLOCK_SIZE];

    sum = warp_reduce_sum(sum);

    __shared__ float warp_sums[BLOCK_SIZE / WARP_SIZE];
    if ((threadIdx.x & (WARP_SIZE - 1)) == 0)
        warp_sums[threadIdx.x / WARP_SIZE] = sum;
    __syncthreads();

    float block_sum = 0.0f;
    if (threadIdx.x < BLOCK_SIZE / WARP_SIZE)
        block_sum = warp_sums[threadIdx.x];

    if (threadIdx.x < WARP_SIZE)
        block_sum = warp_reduce_sum(block_sum);

    if (threadIdx.x == 0)
        atomicAdd(out, block_sum);
}

template <typename F>
float time_ms(F f) {
    cudaEvent_t s,e; CHECK_CUDA(cudaEventCreate(&s)); CHECK_CUDA(cudaEventCreate(&e));
    CHECK_CUDA(cudaEventRecord(s));
    f();
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float ms=0; CHECK_CUDA(cudaEventElapsedTime(&ms,s,e));
    CHECK_CUDA(cudaEventDestroy(s)); CHECK_CUDA(cudaEventDestroy(e));
    return ms;
}

__global__ void fill_kernel(float* x, size_t n, float v) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = v;
}


int main() {
    // Pick sizes small enough that CPU-launch overhead matters (to see graphs shine):
    // Try N≈2^22 first; you can bump to 2^26 to be more bandwidth-bound.
    const size_t N = (size_t)1 << 22;           // ~4M elements
    const int    BLOCK = 256;
    const int    KERNELS_PER_REPLAY = 50;       // number of kernels *inside* the graph
    const int    REPLAYS = 400;                 // number of graph replays
    const int    DIRECT_LOOPS = KERNELS_PER_REPLAY * REPLAYS;

    // Host / Device buffers
    float *dX, *dOut;
    CHECK_CUDA(cudaMalloc(&dX, N * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&dOut, sizeof(float)));

    // Initialize input on device quickly (fill with 1.0f)
    // A tiny kernel just to fill:
    dim3 t(256), g((unsigned)((N + t.x - 1)/t.x));
    fill_kernel<<<g,t>>>(dX, N, 1.0f);
    CHECK_CUDA(cudaDeviceSynchronize());

    dim3 block(BLOCK);
    dim3 grid((int)((N + block.x * 2 - 1) / (block.x * 2)));

    // -------------------------
    // 1) Baseline: direct loops
    // -------------------------
    CHECK_CUDA(cudaMemset(dOut, 0, sizeof(float)));
    float direct_ms = time_ms([&](){
        for (int i=0;i<DIRECT_LOOPS;i++) {
            reduce_shuffle_kernel<BLOCK><<<grid, block>>>(dX, N, dOut);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    });

    float sum_direct=0.0f;
    CHECK_CUDA(cudaMemcpy(&sum_direct, dOut, sizeof(float), cudaMemcpyDeviceToHost));
    printf("[Direct]   time=%7.3f ms, loops=%d, sum=%.1f (expect %.1f)\n",
           direct_ms, DIRECT_LOOPS, sum_direct, (double)N * DIRECT_LOOPS / (KERNELS_PER_REPLAY * REPLAYS));

    // ------------------------------------------------
    // 2) CUDA Graph: capture KERNELS_PER_REPLAY kernels
    // ------------------------------------------------
    CHECK_CUDA(cudaMemset(dOut, 0, sizeof(float)));

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    cudaGraph_t graph;
    cudaGraphExec_t instance;

    // Stream-capture the sequence (KERNELS_PER_REPLAY identical kernel launches)
    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    for (int i=0;i<KERNELS_PER_REPLAY;i++) {
        reduce_shuffle_kernel<BLOCK><<<grid, block, 0, stream>>>(dX, N, dOut);
    }
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&instance, graph, nullptr, nullptr, 0));

    // Warmup a few replays
    for (int i=0;i<5;i++) CHECK_CUDA(cudaGraphLaunch(instance, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    float graph_ms = time_ms([&](){
        for (int r=0; r<REPLAYS; ++r) {
            CHECK_CUDA(cudaGraphLaunch(instance, stream));
        }
        CHECK_CUDA(cudaStreamSynchronize(stream));
    });

    float sum_graph=0.0f;
    CHECK_CUDA(cudaMemcpy(&sum_graph, dOut, sizeof(float), cudaMemcpyDeviceToHost));

    printf("[Graph]    time=%7.3f ms, replays=%d, kernels/replay=%d, total kernels=%d\n",
           graph_ms, REPLAYS, KERNELS_PER_REPLAY, REPLAYS*KERNELS_PER_REPLAY);
    printf("           sum=%.1f (expect %.1f)\n",
           sum_graph, (double)N * REPLAYS);

    // Rough throughput vs total data moved (reads only, lower bound)
    double bytes_total = (double)N * sizeof(float) * (REPLAYS*KERNELS_PER_REPLAY);
    double gbps_direct = bytes_total / (direct_ms/1000.0) / 1e9;
    double gbps_graph  = bytes_total / (graph_ms/1000.0)  / 1e9;
    printf("Throughput lower-bound: Direct ~%.2f GB/s  |  Graph ~%.2f GB/s\n",
           gbps_direct, gbps_graph);

    // Cleanup
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(instance);
    cudaStreamDestroy(stream);
    cudaFree(dX); cudaFree(dOut);
    return 0;
}
