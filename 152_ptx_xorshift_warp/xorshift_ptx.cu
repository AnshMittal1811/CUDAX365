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

__device__ __forceinline__ uint32_t xorshift32_ptx(uint32_t x) {
    uint32_t t;
    asm("shl.b32 %0, %1, 13;" : "=r"(t) : "r"(x));
    asm("xor.b32 %0, %1, %2;" : "=r"(x) : "r"(x), "r"(t));
    asm("shr.u32 %0, %1, 17;" : "=r"(t) : "r"(x));
    asm("xor.b32 %0, %1, %2;" : "=r"(x) : "r"(x), "r"(t));
    asm("shl.b32 %0, %1, 5;" : "=r"(t) : "r"(x));
    asm("xor.b32 %0, %1, %2;" : "=r"(x) : "r"(x), "r"(t));
    return x;
}

__global__ void xorshift_kernel(float *out, int n, uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    uint32_t x = seed ^ static_cast<uint32_t>(idx * 747796405u + 2891336453u);
    x = xorshift32_ptx(x);
    x = xorshift32_ptx(x);
    float v = (x & 0x00FFFFFF) / 16777216.0f;
    out[idx] = v;
}

static void compute_chi2(const std::vector<float> &data) {
    int bins = 64;
    std::vector<int> hist(bins, 0);
    for (float v : data) {
        int bin = static_cast<int>(v * bins);
        if (bin >= bins) {
            bin = bins - 1;
        }
        if (bin < 0) {
            bin = 0;
        }
        hist[bin] += 1;
    }
    double expected = static_cast<double>(data.size()) / bins;
    double chi2 = 0.0;
    for (int count : hist) {
        double diff = count - expected;
        chi2 += diff * diff / expected;
    }
    std::printf("xorshift chi2=%.3f\n", chi2);
}

static void write_binary(const char *path, const std::vector<float> &data) {
    std::FILE *fp = std::fopen(path, "wb");
    if (!fp) {
        std::perror(path);
        return;
    }
    std::fwrite(data.data(), sizeof(float), data.size(), fp);
    std::fclose(fp);
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    if (n <= 0) {
        std::fprintf(stderr, "n must be positive\n");
        return 1;
    }

    float *d_out = nullptr;
    check(cudaMalloc(&d_out, n * sizeof(float)), "cudaMalloc d_out");

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    xorshift_kernel<<<blocks, threads>>>(d_out, n, 1234u);
    check(cudaDeviceSynchronize(), "kernel sync");

    std::vector<float> out(n);
    check(cudaMemcpy(out.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost), "copy to host");
    check(cudaFree(d_out), "cudaFree d_out");

    compute_chi2(out);
    write_binary("xorshift.bin", out);
    std::printf("Wrote xorshift.bin\n");
    return 0;
}
