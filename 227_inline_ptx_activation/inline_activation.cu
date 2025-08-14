#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__device__ __forceinline__ float relu_ptx(float x) {
    float out;
    asm("max.f32 %0, %1, 0f00000000;" : "=f"(out) : "f"(x));
    return out;
}

__global__ void conv_relu(const float *in, const float *kernel, float *out, int w, int h) {
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
            sum += in[iy * w + ix] * kernel[(ky + 1) * 3 + (kx + 1)];
        }
    }
    out[y * w + x] = relu_ptx(sum);
}

int main(int argc, char **argv) {
    int w = (argc > 1) ? std::atoi(argv[1]) : 128;
    int h = (argc > 2) ? std::atoi(argv[2]) : 128;
    size_t total = static_cast<size_t>(w) * h;

    float *d_in = nullptr;
    float *d_k = nullptr;
    float *d_out = nullptr;
    cudaMalloc(&d_in, total * sizeof(float));
    cudaMalloc(&d_k, 9 * sizeof(float));
    cudaMalloc(&d_out, total * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    conv_relu<<<grid, block>>>(d_in, d_k, d_out, w, h);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_k);
    cudaFree(d_out);

    std::printf("inline activation done\n");
    return 0;
}
