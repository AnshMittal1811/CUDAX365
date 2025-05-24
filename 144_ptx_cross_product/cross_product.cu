#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

__device__ __forceinline__ float fma_ptx(float a, float b, float c) {
    float out;
    asm("fma.rn.f32 %0, %1, %2, %3;" : "=f"(out) : "f"(a), "f"(b), "f"(c));
    return out;
}

__global__ void cross_warp_kernel(const float *a, const float *b, float *c, int matrices) {
    int global_thread = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = global_thread / warpSize;
    int lane = threadIdx.x & 31;
    if (warp_id >= matrices) {
        return;
    }
    if (lane >= 9) {
        return;
    }

    int base = warp_id * 9;
    int row = lane / 3;
    int col = lane % 3;
    int row_base = row * 3;

    float a_val = a[base + row_base + col];
    float b_val = b[base + row_base + col];

    unsigned mask = 0x1FF; // first 9 lanes
    float ax = __shfl_sync(mask, a_val, row_base + 0);
    float ay = __shfl_sync(mask, a_val, row_base + 1);
    float az = __shfl_sync(mask, a_val, row_base + 2);
    float bx = __shfl_sync(mask, b_val, row_base + 0);
    float by = __shfl_sync(mask, b_val, row_base + 1);
    float bz = __shfl_sync(mask, b_val, row_base + 2);

    float out = 0.0f;
    if (col == 0) {
        out = fma_ptx(ay, bz, -az * by);
    } else if (col == 1) {
        out = fma_ptx(az, bx, -ax * bz);
    } else {
        out = fma_ptx(ax, by, -ay * bx);
    }

    c[base + row_base + col] = out;
}

static void cross_cpu(const float *a, const float *b, float *c, int matrices) {
    for (int i = 0; i < matrices; ++i) {
        int base = i * 9;
        for (int r = 0; r < 3; ++r) {
            int row = base + r * 3;
            float ax = a[row + 0];
            float ay = a[row + 1];
            float az = a[row + 2];
            float bx = b[row + 0];
            float by = b[row + 1];
            float bz = b[row + 2];
            c[row + 0] = ay * bz - az * by;
            c[row + 1] = az * bx - ax * bz;
            c[row + 2] = ax * by - ay * bx;
        }
    }
}

int main(int argc, char **argv) {
    int matrices = (argc > 1) ? std::atoi(argv[1]) : 4096;
    if (matrices <= 0) {
        std::fprintf(stderr, "matrices must be positive\n");
        return 1;
    }

    size_t total = static_cast<size_t>(matrices) * 9;
    std::vector<float> h_a(total);
    std::vector<float> h_b(total);
    std::vector<float> h_c(total);
    std::vector<float> h_ref(total);

    for (size_t i = 0; i < total; ++i) {
        h_a[i] = static_cast<float>((i % 17) - 8) * 0.1f;
        h_b[i] = static_cast<float>((i % 13) - 6) * 0.07f;
    }

    float *d_a = nullptr;
    float *d_b = nullptr;
    float *d_c = nullptr;
    check(cudaMalloc(&d_a, total * sizeof(float)), "cudaMalloc d_a");
    check(cudaMalloc(&d_b, total * sizeof(float)), "cudaMalloc d_b");
    check(cudaMalloc(&d_c, total * sizeof(float)), "cudaMalloc d_c");
    check(cudaMemcpy(d_a, h_a.data(), total * sizeof(float), cudaMemcpyHostToDevice), "H2D a");
    check(cudaMemcpy(d_b, h_b.data(), total * sizeof(float), cudaMemcpyHostToDevice), "H2D b");

    int threads = 256;
    int warps_per_block = threads / 32;
    int blocks = (matrices + warps_per_block - 1) / warps_per_block;

    cross_warp_kernel<<<blocks, threads>>>(d_a, d_b, d_c, matrices);
    check(cudaDeviceSynchronize(), "kernel sync");

    check(cudaMemcpy(h_c.data(), d_c, total * sizeof(float), cudaMemcpyDeviceToHost), "D2H c");
    cross_cpu(h_a.data(), h_b.data(), h_ref.data(), matrices);

    float max_err = 0.0f;
    for (size_t i = 0; i < total; ++i) {
        float err = std::fabs(h_ref[i] - h_c[i]);
        if (err > max_err) {
            max_err = err;
        }
    }

    std::printf("cross product max_err=%g\n", max_err);

    check(cudaFree(d_a), "cudaFree d_a");
    check(cudaFree(d_b), "cudaFree d_b");
    check(cudaFree(d_c), "cudaFree d_c");
    return 0;
}
