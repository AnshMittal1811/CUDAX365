#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct FlowParams {
    float arm_amp_hot;
    float arm_amp_cold;
    float phi_amp_hot;
    float phi_amp_cold;
    float m_hot;
    float m_cold;
};

static void check_cuda(cudaError_t err, const char *ctx) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", ctx, cudaGetErrorString(err));
        std::exit(1);
    }
}

__global__ void flow_kernel(
    const float *theta,
    const float *omega,
    const float *phi_grav,
    const float *rho_base_hot,
    const float *rho_base_cold,
    float *rho_hot,
    float *phi_hot,
    float *rho_cold,
    float *phi_cold,
    int n,
    float t,
    FlowParams params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) {
        return;
    }
    float phase = theta[idx] - omega[idx] * t;
    float hot_sin = __sinf(params.m_hot * phase);
    float hot_cos = __cosf(params.m_hot * phase);
    float cold_sin = __sinf(params.m_cold * phase);
    float cold_cos = __cosf(params.m_cold * phase);

    float rho_h = rho_base_hot[idx] * (1.0f + params.arm_amp_hot * hot_sin);
    float rho_c = rho_base_cold[idx] * (1.0f + params.arm_amp_cold * cold_sin);
    rho_hot[idx] = fmaxf(rho_h, 0.0f);
    rho_cold[idx] = fmaxf(rho_c, 0.0f);

    phi_hot[idx] = phi_grav[idx] + params.phi_amp_hot * hot_cos;
    phi_cold[idx] = phi_grav[idx] + params.phi_amp_cold * cold_cos;
}

static int parse_int(int argc, char **argv, const char *key, int default_val) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) {
            return std::atoi(argv[i + 1]);
        }
    }
    return default_val;
}

static float parse_float(int argc, char **argv, const char *key, float default_val) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) {
            return std::strtof(argv[i + 1], nullptr);
        }
    }
    return default_val;
}

static std::string parse_string(int argc, char **argv, const char *key, const char *default_val) {
    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], key) == 0) {
            return std::string(argv[i + 1]);
        }
    }
    return std::string(default_val);
}

static void write_frame(const fs::path &path, const float *data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
}

int main(int argc, char **argv) {
    int nx = parse_int(argc, argv, "--nx", 160);
    int ny = parse_int(argc, argv, "--ny", 160);
    int frames = parse_int(argc, argv, "--frames", 1440);
    float dt = parse_float(argc, argv, "--dt", 0.05f);
    std::string out_hot = parse_string(argc, argv, "--out-hot", "frames_hot");
    std::string out_cold = parse_string(argc, argv, "--out-cold", "frames_cold");

    fs::create_directories(out_hot);
    fs::create_directories(out_cold);

    int total = nx * ny;
    size_t bytes = static_cast<size_t>(total) * sizeof(float);

    std::vector<float> theta(total);
    std::vector<float> omega(total);
    std::vector<float> phi_grav(total);
    std::vector<float> rho_base_hot(total);
    std::vector<float> rho_base_cold(total);

    float gm = 1.0f;
    float rs = 0.18f;

    float rho0_hot = 1.2f;
    float r0_hot = 0.45f;
    float sigma_hot = 0.12f;

    float rho0_cold = 0.9f;
    float r0_cold = 0.62f;
    float sigma_cold = 0.16f;

    for (int j = 0; j < ny; ++j) {
        float y = (ny > 1) ? (-1.0f + 2.0f * j / (ny - 1)) : 0.0f;
        for (int i = 0; i < nx; ++i) {
            float x = (nx > 1) ? (-1.0f + 2.0f * i / (nx - 1)) : 0.0f;
            int idx = j * nx + i;
            float r = std::sqrt(x * x + y * y);
            float th = std::atan2(y, x);
            float r_safe = fmaxf(r, rs + 1e-3f);
            float phi = -gm / (r_safe - rs);
            float om = std::sqrt(gm / (r_safe * r_safe * r_safe));

            float rho_hot = rho0_hot * std::exp(-((r - r0_hot) * (r - r0_hot)) / (2.0f * sigma_hot * sigma_hot));
            float rho_cold = rho0_cold * std::exp(-((r - r0_cold) * (r - r0_cold)) / (2.0f * sigma_cold * sigma_cold));
            if (r < rs) {
                rho_hot = 0.0f;
                rho_cold = 0.0f;
            }

            theta[idx] = th;
            omega[idx] = om;
            phi_grav[idx] = phi;
            rho_base_hot[idx] = rho_hot;
            rho_base_cold[idx] = rho_cold;
        }
    }

    float *d_theta = nullptr;
    float *d_omega = nullptr;
    float *d_phi_grav = nullptr;
    float *d_rho_base_hot = nullptr;
    float *d_rho_base_cold = nullptr;
    float *d_rho_hot = nullptr;
    float *d_phi_hot = nullptr;
    float *d_rho_cold = nullptr;
    float *d_phi_cold = nullptr;

    check_cuda(cudaMalloc(&d_theta, bytes), "cudaMalloc theta");
    check_cuda(cudaMalloc(&d_omega, bytes), "cudaMalloc omega");
    check_cuda(cudaMalloc(&d_phi_grav, bytes), "cudaMalloc phi_grav");
    check_cuda(cudaMalloc(&d_rho_base_hot, bytes), "cudaMalloc rho_base_hot");
    check_cuda(cudaMalloc(&d_rho_base_cold, bytes), "cudaMalloc rho_base_cold");
    check_cuda(cudaMalloc(&d_rho_hot, bytes), "cudaMalloc rho_hot");
    check_cuda(cudaMalloc(&d_phi_hot, bytes), "cudaMalloc phi_hot");
    check_cuda(cudaMalloc(&d_rho_cold, bytes), "cudaMalloc rho_cold");
    check_cuda(cudaMalloc(&d_phi_cold, bytes), "cudaMalloc phi_cold");

    check_cuda(cudaMemcpy(d_theta, theta.data(), bytes, cudaMemcpyHostToDevice), "memcpy theta");
    check_cuda(cudaMemcpy(d_omega, omega.data(), bytes, cudaMemcpyHostToDevice), "memcpy omega");
    check_cuda(cudaMemcpy(d_phi_grav, phi_grav.data(), bytes, cudaMemcpyHostToDevice), "memcpy phi_grav");
    check_cuda(cudaMemcpy(d_rho_base_hot, rho_base_hot.data(), bytes, cudaMemcpyHostToDevice), "memcpy rho_base_hot");
    check_cuda(cudaMemcpy(d_rho_base_cold, rho_base_cold.data(), bytes, cudaMemcpyHostToDevice), "memcpy rho_base_cold");

    float *h_rho_hot = nullptr;
    float *h_phi_hot = nullptr;
    float *h_rho_cold = nullptr;
    float *h_phi_cold = nullptr;
    check_cuda(cudaMallocHost(&h_rho_hot, bytes), "cudaMallocHost rho_hot");
    check_cuda(cudaMallocHost(&h_phi_hot, bytes), "cudaMallocHost phi_hot");
    check_cuda(cudaMallocHost(&h_rho_cold, bytes), "cudaMallocHost rho_cold");
    check_cuda(cudaMallocHost(&h_phi_cold, bytes), "cudaMallocHost phi_cold");

    FlowParams params;
    params.arm_amp_hot = 0.35f;
    params.arm_amp_cold = 0.25f;
    params.phi_amp_hot = 0.8f;
    params.phi_amp_cold = 0.6f;
    params.m_hot = 3.0f;
    params.m_cold = 2.0f;

    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    for (int frame = 0; frame < frames; ++frame) {
        float t = frame * dt;
        flow_kernel<<<blocks, threads>>>(
            d_theta, d_omega, d_phi_grav,
            d_rho_base_hot, d_rho_base_cold,
            d_rho_hot, d_phi_hot, d_rho_cold, d_phi_cold,
            total, t, params);
        check_cuda(cudaGetLastError(), "flow_kernel launch");
        check_cuda(cudaDeviceSynchronize(), "flow_kernel sync");

        check_cuda(cudaMemcpy(h_rho_hot, d_rho_hot, bytes, cudaMemcpyDeviceToHost), "memcpy rho_hot");
        check_cuda(cudaMemcpy(h_phi_hot, d_phi_hot, bytes, cudaMemcpyDeviceToHost), "memcpy phi_hot");
        check_cuda(cudaMemcpy(h_rho_cold, d_rho_cold, bytes, cudaMemcpyDeviceToHost), "memcpy rho_cold");
        check_cuda(cudaMemcpy(h_phi_cold, d_phi_cold, bytes, cudaMemcpyDeviceToHost), "memcpy phi_cold");

        char name[64];
        std::snprintf(name, sizeof(name), "rho_%04d.bin", frame);
        write_frame(fs::path(out_hot) / name, h_rho_hot, bytes);
        std::snprintf(name, sizeof(name), "phi_%04d.bin", frame);
        write_frame(fs::path(out_hot) / name, h_phi_hot, bytes);
        std::snprintf(name, sizeof(name), "rho_%04d.bin", frame);
        write_frame(fs::path(out_cold) / name, h_rho_cold, bytes);
        std::snprintf(name, sizeof(name), "phi_%04d.bin", frame);
        write_frame(fs::path(out_cold) / name, h_phi_cold, bytes);

        if (frame % 200 == 0) {
            std::printf("Generated frame %d/%d\n", frame, frames);
        }
    }

    cudaFreeHost(h_rho_hot);
    cudaFreeHost(h_phi_hot);
    cudaFreeHost(h_rho_cold);
    cudaFreeHost(h_phi_cold);

    cudaFree(d_theta);
    cudaFree(d_omega);
    cudaFree(d_phi_grav);
    cudaFree(d_rho_base_hot);
    cudaFree(d_rho_base_cold);
    cudaFree(d_rho_hot);
    cudaFree(d_phi_hot);
    cudaFree(d_rho_cold);
    cudaFree(d_phi_cold);

    return 0;
}
