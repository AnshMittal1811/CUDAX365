#include <cstdio>

__global__ void hand_kernel(float *out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] *= 1.001f;
    }
}

int main() {
    std::printf("hand kernel placeholder\n");
    return 0;
}
