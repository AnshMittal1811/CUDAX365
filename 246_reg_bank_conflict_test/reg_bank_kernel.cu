#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void reg_bank_kernel(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float a = idx * 0.1f;
    float b = idx * 0.2f;
    float c = idx * 0.3f;
    float d = idx * 0.4f;
    float e = idx * 0.5f;
    float f = idx * 0.6f;
    float g = idx * 0.7f;
    float h = idx * 0.8f;
    float sum = a + b + c + d + e + f + g + h;
    out[idx] = sum;
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    float *d_out = nullptr;
    cudaMalloc(&d_out, n * sizeof(float));
    reg_bank_kernel<<<(n + 255) / 256, 256>>>(d_out, n);
    cudaDeviceSynchronize();
    cudaFree(d_out);
    std::printf("reg bank kernel done\n");
    return 0;
}
