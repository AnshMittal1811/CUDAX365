#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

__global__ void denoise_kernel(const float *in, float *out, int width, int height, int pitch, int pad) {
    extern __shared__ float tile[];
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int local_x = threadIdx.x;
    int local_y = threadIdx.y;

    int tile_width = blockDim.x + pad;
    int tile_idx = local_y * tile_width + local_x;

    if (x < width && y < height) {
        tile[tile_idx] = in[y * pitch + x];
    }
    __syncthreads();

    if (x < width && y < height) {
        float v = tile[tile_idx];
        out[y * pitch + x] = v;
    }
}

static float time_kernel(const float *in, float *out, int width, int height, int pitch, int pad) {
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);
    size_t shmem = (block.x + pad) * block.y * sizeof(float);

    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");
    check(cudaEventRecord(start), "record start");
    denoise_kernel<<<grid, block, shmem>>>(in, out, width, height, pitch, pad);
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "sync stop");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    check(cudaEventDestroy(start), "destroy start");
    check(cudaEventDestroy(stop), "destroy stop");
    return ms;
}

int main(int argc, char **argv) {
    int width = (argc > 1) ? std::atoi(argv[1]) : 512;
    int height = (argc > 2) ? std::atoi(argv[2]) : 512;
    if (width <= 0 || height <= 0) {
        std::fprintf(stderr, "invalid dims\n");
        return 1;
    }

    int pitch = width;
    size_t total = static_cast<size_t>(width) * height;

    float *d_in = nullptr;
    float *d_out = nullptr;
    check(cudaMalloc(&d_in, total * sizeof(float)), "cudaMalloc d_in");
    check(cudaMalloc(&d_out, total * sizeof(float)), "cudaMalloc d_out");

    std::FILE *fp = std::fopen("bank_tune.csv", "w");
    if (!fp) {
        std::perror("bank_tune.csv");
        return 1;
    }
    std::fprintf(fp, "pad,ms\n");

    for (int pad = 0; pad <= 8; ++pad) {
        float ms = time_kernel(d_in, d_out, width, height, pitch, pad);
        std::fprintf(fp, "%d,%.6f\n", pad, ms);
    }

    std::fclose(fp);
    check(cudaFree(d_in), "cudaFree d_in");
    check(cudaFree(d_out), "cudaFree d_out");

    std::printf("Wrote bank_tune.csv\n");
    return 0;
}
