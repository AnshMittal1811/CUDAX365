#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ uint32_t xorshift32(uint32_t x) {
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    return x;
}

__global__ void fp8_dropout(uint8_t *data, int n, float drop_prob, uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    uint32_t r = xorshift32(seed ^ idx);
    float v = (r & 0x00FFFFFF) / 16777216.0f;
    if (v < drop_prob) {
        data[idx] = 0;
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 4096;
    uint8_t *d_data = nullptr;
    cudaMalloc(&d_data, n * sizeof(uint8_t));

    fp8_dropout<<<(n + 255) / 256, 256>>>(d_data, n, 0.2f, 1234u);
    cudaDeviceSynchronize();

    cudaFree(d_data);
    std::printf("fp8 dropout done\n");
    return 0;
}
