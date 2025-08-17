#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void int8_conv(const int8_t *in, const int8_t *kernel, int *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    int acc = 0;
    int8_t a = in[idx];
    int8_t b = kernel[idx % 9];
    int packed = (static_cast<int>(a) & 0xFF) | ((static_cast<int>(b) & 0xFF) << 8);
    asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(acc) : "r"(packed), "r"(packed), "r"(acc));
    out[idx] = acc;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int8_t *d_in = nullptr;
    int8_t *d_k = nullptr;
    int *d_out = nullptr;

    cudaMalloc(&d_in, n * sizeof(int8_t));
    cudaMalloc(&d_k, 9 * sizeof(int8_t));
    cudaMalloc(&d_out, n * sizeof(int));

    int8_conv<<<(n + 255) / 256, 256>>>(d_in, d_k, d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_k);
    cudaFree(d_out);

    std::printf("int8 conv done\n");
    return 0;
}
