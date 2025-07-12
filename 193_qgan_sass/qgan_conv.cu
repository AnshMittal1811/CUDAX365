#include <cuda_runtime.h>
#include <cstdio>

__global__ void qgan_conv(float *out, const float *in, const float *kernel, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float sum = 0.0f;
    for (int k = -1; k <= 1; ++k) {
        int ix = min(max(idx + k, 0), n - 1);
        sum += in[ix] * kernel[k + 1];
    }
    out[idx] = sum;
}

int main() {
    int n = 1024;
    float *d_in = nullptr;
    float *d_out = nullptr;
    float *d_k = nullptr;
    cudaMalloc(&d_in, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    cudaMalloc(&d_k, 3 * sizeof(float));

    qgan_conv<<<(n + 255) / 256, 256>>>(d_out, d_in, d_k, n);
    cudaDeviceSynchronize();

    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_k);
    std::printf("qgan_conv done\n");
    return 0;
}
