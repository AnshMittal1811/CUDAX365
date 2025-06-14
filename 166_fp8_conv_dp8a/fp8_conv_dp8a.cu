#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

__global__ void conv_int8_dp4a(const int8_t *in, const int8_t *kernel, int *out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    int sum = 0;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int ix = min(max(x + kx, 0), w - 1);
            int iy = min(max(y + ky, 0), h - 1);
            int in_idx = iy * w + ix;
            int k_idx = (ky + 1) * 3 + (kx + 1);
            int8_t a = in[in_idx];
            int8_t b = kernel[k_idx];
            int packed = (static_cast<int>(a) & 0xFF) | ((static_cast<int>(b) & 0xFF) << 8);
            int acc = 0;
            asm("dp4a.s32.s32 %0, %1, %2, %3;" : "=r"(acc) : "r"(packed), "r"(packed), "r"(acc));
            sum += acc;
        }
    }
    out[y * w + x] = sum;
}

__global__ void conv_fp16(const half *in, const half *kernel, float *out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }

    float sum = 0.0f;
    for (int ky = -1; ky <= 1; ++ky) {
        for (int kx = -1; kx <= 1; ++kx) {
            int ix = min(max(x + kx, 0), w - 1);
            int iy = min(max(y + ky, 0), h - 1);
            int in_idx = iy * w + ix;
            int k_idx = (ky + 1) * 3 + (kx + 1);
            sum += __half2float(in[in_idx]) * __half2float(kernel[k_idx]);
        }
    }
    out[y * w + x] = sum;
}

int main(int argc, char **argv) {
    int w = (argc > 1) ? std::atoi(argv[1]) : 256;
    int h = (argc > 2) ? std::atoi(argv[2]) : 256;
    if (w <= 0 || h <= 0) {
        std::fprintf(stderr, "invalid dims\n");
        return 1;
    }

    size_t total = static_cast<size_t>(w) * h;

    int8_t *d_in8 = nullptr;
    int8_t *d_k8 = nullptr;
    int *d_out8 = nullptr;
    check(cudaMalloc(&d_in8, total * sizeof(int8_t)), "malloc d_in8");
    check(cudaMalloc(&d_k8, 9 * sizeof(int8_t)), "malloc d_k8");
    check(cudaMalloc(&d_out8, total * sizeof(int)), "malloc d_out8");

    half *d_in16 = nullptr;
    half *d_k16 = nullptr;
    float *d_out16 = nullptr;
    check(cudaMalloc(&d_in16, total * sizeof(half)), "malloc d_in16");
    check(cudaMalloc(&d_k16, 9 * sizeof(half)), "malloc d_k16");
    check(cudaMalloc(&d_out16, total * sizeof(float)), "malloc d_out16");

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);

    conv_int8_dp4a<<<grid, block>>>(d_in8, d_k8, d_out8, w, h);
    conv_fp16<<<grid, block>>>(d_in16, d_k16, d_out16, w, h);
    check(cudaDeviceSynchronize(), "kernel sync");

    std::printf("Ran int8 dp4a and fp16 conv kernels\n");

    check(cudaFree(d_in8), "free d_in8");
    check(cudaFree(d_k8), "free d_k8");
    check(cudaFree(d_out8), "free d_out8");
    check(cudaFree(d_in16), "free d_in16");
    check(cudaFree(d_k16), "free d_k16");
    check(cudaFree(d_out16), "free d_out16");
    return 0;
}
