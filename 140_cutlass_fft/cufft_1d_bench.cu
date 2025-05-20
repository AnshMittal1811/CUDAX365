#include <cuda_runtime.h>
#include <cufft.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

static void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

static void check_cufft(cufftResult status, const char *msg) {
    if (status != CUFFT_SUCCESS) {
        std::fprintf(stderr, "cuFFT error %s: %d\n", msg, status);
        std::exit(1);
    }
}

static void cpu_dft(const std::vector<cufftComplex> &in, std::vector<cufftComplex> &out, int n) {
    const float kTwoPi = 6.283185307179586f;
    for (int k = 0; k < n; ++k) {
        float sum_re = 0.0f;
        float sum_im = 0.0f;
        for (int t = 0; t < n; ++t) {
            float angle = -kTwoPi * static_cast<float>(k * t) / static_cast<float>(n);
            float c = std::cos(angle);
            float s = std::sin(angle);
            float re = in[t].x;
            float im = in[t].y;
            sum_re += re * c - im * s;
            sum_im += re * s + im * c;
        }
        out[k].x = sum_re;
        out[k].y = sum_im;
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : 4096;
    int batch = (argc > 2) ? std::atoi(argv[2]) : 64;
    if (n <= 0 || batch <= 0) {
        std::fprintf(stderr, "n and batch must be positive\n");
        return 1;
    }

    size_t total = static_cast<size_t>(n) * static_cast<size_t>(batch);
    std::vector<cufftComplex> h_in(total);
    std::vector<cufftComplex> h_out(total);

    std::mt19937 rng(0);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (size_t i = 0; i < total; ++i) {
        h_in[i].x = dist(rng);
        h_in[i].y = dist(rng);
    }

    cufftComplex *d_in = nullptr;
    cufftComplex *d_out = nullptr;
    check_cuda(cudaMalloc(&d_in, total * sizeof(cufftComplex)), "cudaMalloc d_in");
    check_cuda(cudaMalloc(&d_out, total * sizeof(cufftComplex)), "cudaMalloc d_out");
    check_cuda(cudaMemcpy(d_in, h_in.data(), total * sizeof(cufftComplex), cudaMemcpyHostToDevice),
               "cudaMemcpy H2D");

    cufftHandle plan;
    check_cufft(cufftPlan1d(&plan, n, CUFFT_C2C, batch), "cufftPlan1d");

    check_cufft(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD), "cufftExecC2C warmup");
    check_cuda(cudaDeviceSynchronize(), "sync warmup");

    cudaEvent_t start, stop;
    check_cuda(cudaEventCreate(&start), "event create start");
    check_cuda(cudaEventCreate(&stop), "event create stop");

    int iters = 50;
    check_cuda(cudaEventRecord(start), "event record start");
    for (int i = 0; i < iters; ++i) {
        check_cufft(cufftExecC2C(plan, d_in, d_out, CUFFT_FORWARD), "cufftExecC2C");
    }
    check_cuda(cudaEventRecord(stop), "event record stop");
    check_cuda(cudaEventSynchronize(stop), "event sync stop");

    float ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&ms, start, stop), "event elapsed");
    float avg_ms = ms / static_cast<float>(iters);

    check_cuda(cudaMemcpy(h_out.data(), d_out, total * sizeof(cufftComplex), cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H");

    int check_n = n;
    if (check_n > 2048) {
        check_n = 2048;
    }
    std::vector<cufftComplex> cpu_in(check_n);
    std::vector<cufftComplex> cpu_out(check_n);
    for (int i = 0; i < check_n; ++i) {
        cpu_in[i] = h_in[i];
    }
    cpu_dft(cpu_in, cpu_out, check_n);

    float max_err = 0.0f;
    for (int i = 0; i < check_n; ++i) {
        float dx = cpu_out[i].x - h_out[i].x;
        float dy = cpu_out[i].y - h_out[i].y;
        float err = std::sqrt(dx * dx + dy * dy);
        if (err > max_err) {
            max_err = err;
        }
    }

    std::printf("cuFFT 1D C2C n=%d batch=%d avg_ms=%.4f max_err(first %d)=%g\n",
                n, batch, avg_ms, check_n, max_err);

    check_cufft(cufftDestroy(plan), "cufftDestroy");
    check_cuda(cudaFree(d_in), "cudaFree d_in");
    check_cuda(cudaFree(d_out), "cudaFree d_out");
    check_cuda(cudaEventDestroy(start), "event destroy start");
    check_cuda(cudaEventDestroy(stop), "event destroy stop");

    return 0;
}
