#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

__device__ __forceinline__ float add_ptx(float a, float b) {
    float out;
    asm("add.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float sub_ptx(float a, float b) {
    float out;
    asm("add.rn.f32 %0, %1, -%2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__global__ void haar_wavelet(const float *in, float *approx, float *detail, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int i = idx * 2;
    if (i + 1 >= n) {
        return;
    }
    float a = in[i];
    float b = in[i + 1];
    approx[idx] = add_ptx(a, b) * 0.5f;
    detail[idx] = sub_ptx(a, b) * 0.5f;
}

static void haar_cpu(const std::vector<float> &in, std::vector<float> &approx, std::vector<float> &detail) {
    for (size_t i = 0; i + 1 < in.size(); i += 2) {
        float a = in[i];
        float b = in[i + 1];
        approx[i / 2] = 0.5f * (a + b);
        detail[i / 2] = 0.5f * (a - b);
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 1 << 20;
    if (n <= 0 || (n % 2) != 0) {
        std::fprintf(stderr, "n must be positive and even\n");
        return 1;
    }

    std::vector<float> h_in(n);
    for (int i = 0; i < n; ++i) {
        h_in[i] = static_cast<float>(i % 256) / 255.0f;
    }

    std::vector<float> h_approx(n / 2);
    std::vector<float> h_detail(n / 2);
    std::vector<float> h_approx_ref(n / 2);
    std::vector<float> h_detail_ref(n / 2);

    float *d_in = nullptr;
    float *d_approx = nullptr;
    float *d_detail = nullptr;
    check(cudaMalloc(&d_in, n * sizeof(float)), "cudaMalloc d_in");
    check(cudaMalloc(&d_approx, (n / 2) * sizeof(float)), "cudaMalloc d_approx");
    check(cudaMalloc(&d_detail, (n / 2) * sizeof(float)), "cudaMalloc d_detail");
    check(cudaMemcpy(d_in, h_in.data(), n * sizeof(float), cudaMemcpyHostToDevice), "H2D in");

    int threads = 256;
    int blocks = (n / 2 + threads - 1) / threads;
    haar_wavelet<<<blocks, threads>>>(d_in, d_approx, d_detail, n);
    check(cudaDeviceSynchronize(), "kernel sync");

    check(cudaMemcpy(h_approx.data(), d_approx, (n / 2) * sizeof(float), cudaMemcpyDeviceToHost), "D2H approx");
    check(cudaMemcpy(h_detail.data(), d_detail, (n / 2) * sizeof(float), cudaMemcpyDeviceToHost), "D2H detail");

    haar_cpu(h_in, h_approx_ref, h_detail_ref);

    float max_err = 0.0f;
    for (int i = 0; i < n / 2; ++i) {
        float err_a = std::abs(h_approx[i] - h_approx_ref[i]);
        float err_d = std::abs(h_detail[i] - h_detail_ref[i]);
        if (err_a > max_err) {
            max_err = err_a;
        }
        if (err_d > max_err) {
            max_err = err_d;
        }
    }

    std::printf("Haar wavelet max_err=%g\n", max_err);

    check(cudaFree(d_in), "cudaFree d_in");
    check(cudaFree(d_approx), "cudaFree d_approx");
    check(cudaFree(d_detail), "cudaFree d_detail");
    return 0;
}
