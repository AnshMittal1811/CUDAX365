#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void baseline_conv(const float *in, const float *kernel, float *out, int w, int h) {
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
    out[y * w + x] = sum;
}

int main() {
    std::printf("baseline conv placeholder\n");
    return 0;
}
