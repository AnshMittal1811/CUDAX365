#include <cuda_runtime.h>

#include <cstdio>

__global__ void fp6_emulated(const int8_t *in, int8_t *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int8_t v = in[idx];
        out[idx] = v;
    }
}

int main() {
    std::printf("fp6 emulation kernel\n");
    return 0;
}
