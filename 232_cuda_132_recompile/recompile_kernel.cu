#include <cstdio>

__global__ void recompile_kernel(float *out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        out[0] = 1.0f;
    }
}

int main() {
    std::printf("recompile kernel\n");
    return 0;
}
