#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

__global__ void occ_kernel(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float v = static_cast<float>(idx) * 0.0001f;
        v = v * 1.001f + 0.01f;
        out[idx] = v;
    }
}

static float time_kernel(float *d_out, int n, int block) {
    int grid = (n + block - 1) / block;
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");
    check(cudaEventRecord(start), "record start");
    occ_kernel<<<grid, block>>>(d_out, n);
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "sync stop");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    check(cudaEventDestroy(start), "destroy start");
    check(cudaEventDestroy(stop), "destroy stop");
    return ms;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 24);
    int min_block = (argc > 2) ? std::atoi(argv[2]) : 64;
    int max_block = (argc > 3) ? std::atoi(argv[3]) : 1024;
    int step = (argc > 4) ? std::atoi(argv[4]) : 32;

    if (n <= 0 || min_block <= 0 || max_block <= 0 || step <= 0) {
        std::fprintf(stderr, "invalid inputs\n");
        return 1;
    }

    float *d_out = nullptr;
    check(cudaMalloc(&d_out, n * sizeof(float)), "cudaMalloc d_out");

    cudaDeviceProp prop{};
    check(cudaGetDeviceProperties(&prop, 0), "cudaGetDeviceProperties");
    int max_threads_sm = prop.maxThreadsPerMultiProcessor;

    std::FILE *fp = std::fopen("occupancy_results.csv", "w");
    if (!fp) {
        std::perror("occupancy_results.csv");
        return 1;
    }
    std::fprintf(fp, "block,active_blocks,occupancy,ms\n");

    for (int block = min_block; block <= max_block; block += step) {
        int active_blocks = 0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&active_blocks, occ_kernel, block, 0),
              "cudaOccupancyMaxActiveBlocksPerMultiprocessor");
        float occupancy = (active_blocks * block) / static_cast<float>(max_threads_sm);
        float ms = time_kernel(d_out, n, block);
        std::fprintf(fp, "%d,%d,%.4f,%.6f\n", block, active_blocks, occupancy, ms);
    }

    std::fclose(fp);
    check(cudaFree(d_out), "cudaFree d_out");

    std::printf("Wrote occupancy_results.csv\n");
    return 0;
}
