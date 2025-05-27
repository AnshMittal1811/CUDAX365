#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

__global__ void bank_kernel(float *out, int stride, int iters) {
    extern __shared__ float sh[];
    int tid = threadIdx.x;
    sh[tid] = static_cast<float>(tid);
    __syncthreads();

    float v = 0.0f;
    int mask = blockDim.x - 1;
    for (int i = 0; i < iters; ++i) {
        int idx = (tid * stride) & mask;
        v += sh[idx];
    }
    out[blockIdx.x * blockDim.x + tid] = v;
}

static float time_kernel(float *out, int blocks, int threads, int stride, int iters) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");
    check(cudaEventRecord(start), "record start");
    bank_kernel<<<blocks, threads, threads * sizeof(float)>>>(out, stride, iters);
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "sync stop");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    check(cudaEventDestroy(start), "destroy start");
    check(cudaEventDestroy(stop), "destroy stop");
    return ms;
}

int main(int argc, char **argv) {
    int threads = (argc > 1) ? std::atoi(argv[1]) : 256;
    int blocks = (argc > 2) ? std::atoi(argv[2]) : 256;
    int iters = (argc > 3) ? std::atoi(argv[3]) : 1000;
    if (threads <= 0 || blocks <= 0 || iters <= 0) {
        std::fprintf(stderr, "invalid inputs\n");
        return 1;
    }
    if ((threads & (threads - 1)) != 0) {
        std::fprintf(stderr, "threads must be a power of two\n");
        return 1;
    }

    float *d_out = nullptr;
    check(cudaMalloc(&d_out, blocks * threads * sizeof(float)), "cudaMalloc d_out");

    std::FILE *fp = std::fopen("bank_conflicts.csv", "w");
    if (!fp) {
        std::perror("bank_conflicts.csv");
        return 1;
    }
    std::fprintf(fp, "stride,ms\n");

    for (int stride = 1; stride <= 32; ++stride) {
        float ms = time_kernel(d_out, blocks, threads, stride, iters);
        std::fprintf(fp, "%d,%.6f\n", stride, ms);
    }

    std::fclose(fp);
    check(cudaFree(d_out), "cudaFree d_out");
    std::printf("Wrote bank_conflicts.csv\n");
    return 0;
}
