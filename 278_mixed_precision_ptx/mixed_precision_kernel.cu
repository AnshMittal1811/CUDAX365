#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include <cstdio>

__global__ void mixed_kernel(const half *in, const int8_t *in8, half *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        half v = in[idx];
        int8_t q = in8[idx];
        out[idx] = __hadd(v, __float2half(static_cast<float>(q) * 0.01f));
    }
}

int main() {
    std::printf("mixed precision kernel placeholder\n");
    return 0;
}
