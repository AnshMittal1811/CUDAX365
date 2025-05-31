#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

static void check_cuda(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

static void check_curand(curandStatus_t status, const char *msg) {
    if (status != CURAND_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuRAND error %s: %d\n", msg, status);
        std::exit(1);
    }
}

__global__ void sobol_device_kernel(float *out, int n, curandDirectionVectors32_t *dirs) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    curandStateScrambledSobol32_t state;
    curand_init(dirs, idx, 0, &state);
    out[idx] = curand_uniform(&state);
}

static void compute_stats(const std::vector<float> &data, const char *label) {
    double sum = 0.0;
    double sum_sq = 0.0;
    int bins = 64;
    std::vector<int> hist(bins, 0);

    for (float v : data) {
        sum += v;
        sum_sq += v * v;
        int bin = static_cast<int>(v * bins);
        if (bin >= bins) {
            bin = bins - 1;
        }
        if (bin < 0) {
            bin = 0;
        }
        hist[bin] += 1;
    }

    double mean = sum / data.size();
    double var = sum_sq / data.size() - mean * mean;

    double expected = static_cast<double>(data.size()) / bins;
    double chi2 = 0.0;
    for (int count : hist) {
        double diff = count - expected;
        chi2 += diff * diff / expected;
    }

    std::printf("%s mean=%.6f var=%.6f chi2=%.3f\n", label, mean, var, chi2);
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

    std::vector<float> host_sobol(n);
    std::vector<float> device_sobol(n);

    curandGenerator_t host_gen;
    check_curand(curandCreateGeneratorHost(&host_gen, CURAND_RNG_QUASI_SOBOL32), "create host gen");
    check_curand(curandSetQuasiRandomGeneratorDimensions(host_gen, 1), "set dims");
    check_curand(curandGenerateUniform(host_gen, host_sobol.data(), n), "generate host sobol");
    check_curand(curandDestroyGenerator(host_gen), "destroy host gen");

    float *d_out = nullptr;
    check_cuda(cudaMalloc(&d_out, n * sizeof(float)), "cudaMalloc d_out");

    curandDirectionVectors32_t *d_dirs = nullptr;
    check_curand(curandGetDirectionVectors32(&d_dirs, CURAND_SCRAMBLED_DIRECTION_VECTORS_32_JOEKUO6),
                 "get direction vectors");

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    sobol_device_kernel<<<blocks, threads>>>(d_out, n, d_dirs);
    check_cuda(cudaDeviceSynchronize(), "device sync");

    check_cuda(cudaMemcpy(device_sobol.data(), d_out, n * sizeof(float), cudaMemcpyDeviceToHost),
               "copy to host");
    check_cuda(cudaFree(d_out), "cudaFree d_out");

    compute_stats(host_sobol, "host_sobol");
    compute_stats(device_sobol, "device_sobol");

    write_binary("sobol_host.bin", host_sobol);
    write_binary("sobol_device.bin", device_sobol);

    std::printf("Wrote sobol_host.bin and sobol_device.bin\n");
    return 0;
}
