#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

constexpr float kPi = 3.14159265358979323846f;

struct FlowParams {
    float rho0;
    float r0;
    float sigma;
    float arm_amp;
    float arm_m;
    float omega;
    float mag_amp;
    float mag_m;
    float spin_scale;
};

struct GlobalParams {
    float gm_smbh;
    float gm_bh;
    float rs_smbh;
    float rs_bh;
    float lens_strength;
};

struct OrbitParams {
    float a0;
    float e;
    float inc;
    float node;
    float phase;
    float omega0;
    float a_min;
    float tau;
};

static void check_cuda(cudaError_t err, const char *ctx) {
    if (err != cudaSuccess) {
        std::fprintf(stderr, "CUDA error at %s: %s\n", ctx, cudaGetErrorString(err));
        std::exit(1);
    }
}

__device__ __forceinline__ float3 rotate_local(const float3 p, float spin, const float4 orient) {
    float cs = __cosf(spin);
    float sn = __sinf(spin);
    float x = p.x * cs - p.y * sn;
    float y = p.x * sn + p.y * cs;
    float z = p.z;

    float cosi = orient.x;
    float sini = orient.y;
    float cosn = orient.z;
    float sinn = orient.w;

    float y1 = y * cosi - z * sini;
    float z1 = y * sini + z * cosi;

    float X = cosn * x - sinn * y1;
    float Y = sinn * x + cosn * y1;
    float Z = z1;

    return make_float3(X, Y, Z);
}

__global__ __launch_bounds__(256)
void generate_flow(
    const float3 *bh_pos,
    const float3 *bh_tan,
    const float4 *orient,
    const float *spin_offset,
    const float3 *base_pos,
    const float *base_r,
    const float *base_theta,
    float *out,
    int points_per_bh,
    int n_bh,
    float t,
    FlowParams flow,
    GlobalParams global) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = points_per_bh * n_bh;
    if (idx >= total) {
        return;
    }
    int bh = idx / points_per_bh;
    int local = idx - bh * points_per_bh;

    float spin = flow.omega * t + spin_offset[bh] * flow.spin_scale;
    float3 local_pos = rotate_local(base_pos[local], spin, orient[bh]);

    float3 world = make_float3(
        local_pos.x + bh_pos[bh].x,
        local_pos.y + bh_pos[bh].y,
        local_pos.z + bh_pos[bh].z);

    float r_local = sqrtf(local_pos.x * local_pos.x + local_pos.y * local_pos.y + local_pos.z * local_pos.z);
    float r_smbh = sqrtf(world.x * world.x + world.y * world.y + world.z * world.z);
    r_local = fmaxf(r_local, global.rs_bh + 1e-3f);
    r_smbh = fmaxf(r_smbh, global.rs_smbh + 1e-3f);

    float theta = base_theta[local];
    float rho = flow.rho0 * __expf(-((base_r[local] - flow.r0) * (base_r[local] - flow.r0)) /
                                  (2.0f * flow.sigma * flow.sigma));
    float spiral = 1.0f + flow.arm_amp * __sinf(flow.arm_m * (theta - flow.omega * t));
    float magnetic = 1.0f + flow.mag_amp * __cosf(flow.mag_m * theta + 0.5f * flow.omega * t);
    rho = rho * spiral * magnetic;

    float phi = -global.gm_smbh / (r_smbh - global.rs_smbh) - global.gm_bh / (r_local - global.rs_bh);

    float lens = 1.0f + global.lens_strength *
        (global.rs_smbh * global.rs_smbh / (r_smbh * r_smbh + 1e-3f) +
         global.rs_bh * global.rs_bh / (r_local * r_local + 1e-3f));

    float v_mag = sqrtf(global.gm_smbh / r_smbh);
    v_mag = fminf(v_mag, 0.7f);
    float beta = v_mag;
    float gamma = rsqrtf(fmaxf(1.0f - beta * beta, 1e-3f));
    float3 tan = bh_tan[bh];
    float tan_mag = sqrtf(tan.x * tan.x + tan.y * tan.y + tan.z * tan.z);
    float mu = (tan_mag > 1e-5f) ? (tan.z / tan_mag) : 0.0f;
    float doppler = 1.0f / (gamma * (1.0f - beta * mu));
    float grav = sqrtf(fmaxf(1.0f - global.rs_smbh / r_smbh, 0.15f));

    float intensity = lens * doppler * grav;
    intensity = fminf(fmaxf(intensity, 0.4f), 1.8f);

    int out_idx = idx * 6;
    out[out_idx + 0] = world.x;
    out[out_idx + 1] = world.y;
    out[out_idx + 2] = world.z;
    out[out_idx + 3] = rho;
    out[out_idx + 4] = phi;
    out[out_idx + 5] = intensity;
}

static float rand_uniform(std::mt19937 &rng, float lo, float hi) {
    std::uniform_real_distribution<float> dist(lo, hi);
    return dist(rng);
}

static void compute_orbit_state(const OrbitParams &orb, float t, float3 &pos, float3 &tan) {
    float a = orb.a_min + (orb.a0 - orb.a_min) / (1.0f + t / orb.tau);
    float theta = orb.omega0 * t + orb.phase;
    float r = a * (1.0f - orb.e * orb.e) / (1.0f + orb.e * std::cos(theta));

    float x = r * std::cos(theta);
    float y = r * std::sin(theta);
    float tx = -std::sin(theta);
    float ty = std::cos(theta);

    float cosi = std::cos(orb.inc);
    float sini = std::sin(orb.inc);
    float cosn = std::cos(orb.node);
    float sinn = std::sin(orb.node);

    float y1 = y * cosi;
    float z1 = y * sini;
    float ty1 = ty * cosi;
    float tz1 = ty * sini;

    pos = make_float3(cosn * x - sinn * y1, sinn * x + cosn * y1, z1);
    tan = make_float3(cosn * tx - sinn * ty1, sinn * tx + cosn * ty1, tz1);
}

static void write_frame(const fs::path &path, const float *data, size_t bytes) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char *>(data), static_cast<std::streamsize>(bytes));
}

int main(int argc, char **argv) {
    int n_bh = 8;
    int points_per_bh = 384;
    int frames = 1200;
    float dt = 0.05f;
    std::string out_dir = "frames_multi";

    for (int i = 1; i + 1 < argc; ++i) {
        if (std::strcmp(argv[i], "--n-bh") == 0) {
            n_bh = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--points-per-bh") == 0) {
            points_per_bh = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--frames") == 0) {
            frames = std::atoi(argv[i + 1]);
        } else if (std::strcmp(argv[i], "--dt") == 0) {
            dt = std::strtof(argv[i + 1], nullptr);
        } else if (std::strcmp(argv[i], "--out") == 0) {
            out_dir = argv[i + 1];
        }
    }

    if (n_bh != 8) {
        std::fprintf(stderr, "This kernel expects n_bh=8 (binary + 6 singles).\n");
        return 1;
    }

    fs::path base_dir(out_dir);
    fs::path hot_dir = base_dir / "hot";
    fs::path cold_dir = base_dir / "cold";
    fs::path bh_dir = base_dir / "bh";
    fs::create_directories(hot_dir);
    fs::create_directories(cold_dir);
    fs::create_directories(bh_dir);

    std::mt19937 rng(19);

    OrbitParams binary_com;
    binary_com.a0 = 0.8f;
    binary_com.e = 0.18f;
    binary_com.inc = 0.15f;
    binary_com.node = 0.4f;
    binary_com.phase = 0.2f;
    binary_com.omega0 = 1.6f;
    binary_com.a_min = 0.35f;
    binary_com.tau = 22.0f;

    float bin_sep = 0.18f;
    float bin_omega = 6.0f;
    float bin_phase = 0.0f;

    std::vector<OrbitParams> singles;
    for (int i = 0; i < 6; ++i) {
        OrbitParams o;
        o.a0 = rand_uniform(rng, 0.9f, 1.6f);
        o.e = rand_uniform(rng, 0.0f, 0.3f);
        o.inc = rand_uniform(rng, 0.0f, 0.5f);
        o.node = rand_uniform(rng, 0.0f, 2.0f * kPi);
        o.phase = rand_uniform(rng, 0.0f, 2.0f * kPi);
        o.omega0 = rand_uniform(rng, 0.9f, 1.3f);
        o.a_min = 0.45f;
        o.tau = 24.0f;
        singles.push_back(o);
    }

    std::vector<float4> orient(n_bh);
    std::vector<float> spin_offset(n_bh);

    auto set_orient = [&](int idx, float inc, float node) {
        orient[idx] = make_float4(std::cos(inc), std::sin(inc), std::cos(node), std::sin(node));
        spin_offset[idx] = rand_uniform(rng, 0.0f, 2.0f * kPi);
    };

    set_orient(0, binary_com.inc, binary_com.node);
    set_orient(1, binary_com.inc, binary_com.node);
    for (int i = 0; i < 6; ++i) {
        set_orient(i + 2, singles[i].inc, singles[i].node);
    }

    std::vector<float3> hot_base(points_per_bh);
    std::vector<float3> cold_base(points_per_bh);
    std::vector<float> hot_r(points_per_bh);
    std::vector<float> cold_r(points_per_bh);
    std::vector<float> hot_theta(points_per_bh);
    std::vector<float> cold_theta(points_per_bh);

    std::normal_distribution<float> hot_r_dist(0.12f, 0.04f);
    std::normal_distribution<float> cold_r_dist(0.2f, 0.06f);
    std::normal_distribution<float> hot_z_dist(0.0f, 0.02f);
    std::normal_distribution<float> cold_z_dist(0.0f, 0.015f);
    std::uniform_real_distribution<float> angle_dist(0.0f, 2.0f * kPi);

    for (int i = 0; i < points_per_bh; ++i) {
        float r = fmaxf(hot_r_dist(rng), 0.04f);
        float theta = angle_dist(rng);
        float z = hot_z_dist(rng);
        hot_base[i] = make_float3(r * std::cos(theta), r * std::sin(theta), z);
        hot_r[i] = r;
        hot_theta[i] = theta;

        r = fmaxf(cold_r_dist(rng), 0.05f);
        theta = angle_dist(rng);
        z = cold_z_dist(rng);
        cold_base[i] = make_float3(r * std::cos(theta), r * std::sin(theta), z);
        cold_r[i] = r;
        cold_theta[i] = theta;
    }

    FlowParams hot;
    hot.rho0 = 1.1f;
    hot.r0 = 0.12f;
    hot.sigma = 0.05f;
    hot.arm_amp = 0.4f;
    hot.arm_m = 3.0f;
    hot.omega = 5.2f;
    hot.mag_amp = 0.2f;
    hot.mag_m = 4.0f;
    hot.spin_scale = 0.8f;

    FlowParams cold;
    cold.rho0 = 0.85f;
    cold.r0 = 0.2f;
    cold.sigma = 0.07f;
    cold.arm_amp = 0.25f;
    cold.arm_m = 2.0f;
    cold.omega = 3.6f;
    cold.mag_amp = 0.15f;
    cold.mag_m = 3.0f;
    cold.spin_scale = 0.6f;

    GlobalParams global;
    global.gm_smbh = 28.0f;
    global.gm_bh = 0.7f;
    global.rs_smbh = 0.28f;
    global.rs_bh = 0.05f;
    global.lens_strength = 0.6f;

    int total_points = n_bh * points_per_bh;
    size_t out_bytes = static_cast<size_t>(total_points) * 6 * sizeof(float);

    float3 *d_hot_base = nullptr;
    float3 *d_cold_base = nullptr;
    float *d_hot_r = nullptr;
    float *d_cold_r = nullptr;
    float *d_hot_theta = nullptr;
    float *d_cold_theta = nullptr;
    float3 *d_bh_pos = nullptr;
    float3 *d_bh_tan = nullptr;
    float4 *d_orient = nullptr;
    float *d_spin_offset = nullptr;
    float *d_hot_out = nullptr;
    float *d_cold_out = nullptr;

    check_cuda(cudaMalloc(&d_hot_base, sizeof(float3) * points_per_bh), "cudaMalloc hot_base");
    check_cuda(cudaMalloc(&d_cold_base, sizeof(float3) * points_per_bh), "cudaMalloc cold_base");
    check_cuda(cudaMalloc(&d_hot_r, sizeof(float) * points_per_bh), "cudaMalloc hot_r");
    check_cuda(cudaMalloc(&d_cold_r, sizeof(float) * points_per_bh), "cudaMalloc cold_r");
    check_cuda(cudaMalloc(&d_hot_theta, sizeof(float) * points_per_bh), "cudaMalloc hot_theta");
    check_cuda(cudaMalloc(&d_cold_theta, sizeof(float) * points_per_bh), "cudaMalloc cold_theta");
    check_cuda(cudaMalloc(&d_bh_pos, sizeof(float3) * n_bh), "cudaMalloc bh_pos");
    check_cuda(cudaMalloc(&d_bh_tan, sizeof(float3) * n_bh), "cudaMalloc bh_tan");
    check_cuda(cudaMalloc(&d_orient, sizeof(float4) * n_bh), "cudaMalloc orient");
    check_cuda(cudaMalloc(&d_spin_offset, sizeof(float) * n_bh), "cudaMalloc spin_offset");
    check_cuda(cudaMalloc(&d_hot_out, out_bytes), "cudaMalloc hot_out");
    check_cuda(cudaMalloc(&d_cold_out, out_bytes), "cudaMalloc cold_out");

    check_cuda(cudaMemcpy(d_hot_base, hot_base.data(), sizeof(float3) * points_per_bh, cudaMemcpyHostToDevice), "copy hot_base");
    check_cuda(cudaMemcpy(d_cold_base, cold_base.data(), sizeof(float3) * points_per_bh, cudaMemcpyHostToDevice), "copy cold_base");
    check_cuda(cudaMemcpy(d_hot_r, hot_r.data(), sizeof(float) * points_per_bh, cudaMemcpyHostToDevice), "copy hot_r");
    check_cuda(cudaMemcpy(d_cold_r, cold_r.data(), sizeof(float) * points_per_bh, cudaMemcpyHostToDevice), "copy cold_r");
    check_cuda(cudaMemcpy(d_hot_theta, hot_theta.data(), sizeof(float) * points_per_bh, cudaMemcpyHostToDevice), "copy hot_theta");
    check_cuda(cudaMemcpy(d_cold_theta, cold_theta.data(), sizeof(float) * points_per_bh, cudaMemcpyHostToDevice), "copy cold_theta");
    check_cuda(cudaMemcpy(d_orient, orient.data(), sizeof(float4) * n_bh, cudaMemcpyHostToDevice), "copy orient");
    check_cuda(cudaMemcpy(d_spin_offset, spin_offset.data(), sizeof(float) * n_bh, cudaMemcpyHostToDevice), "copy spin_offset");

    float *h_hot_out = nullptr;
    float *h_cold_out = nullptr;
    check_cuda(cudaMallocHost(&h_hot_out, out_bytes), "cudaMallocHost hot_out");
    check_cuda(cudaMallocHost(&h_cold_out, out_bytes), "cudaMallocHost cold_out");

    std::vector<float3> bh_pos(n_bh);
    std::vector<float3> bh_tan(n_bh);

    dim3 block(256);
    dim3 grid((total_points + block.x - 1) / block.x);

    for (int frame = 0; frame < frames; ++frame) {
        float t = frame * dt;

        float3 com_pos, com_tan;
        compute_orbit_state(binary_com, t, com_pos, com_tan);
        float bin_phase_t = bin_phase + bin_omega * t;
        float dx = 0.5f * bin_sep * std::cos(bin_phase_t);
        float dy = 0.5f * bin_sep * std::sin(bin_phase_t);

        float cosi = std::cos(binary_com.inc);
        float sini = std::sin(binary_com.inc);
        float cosn = std::cos(binary_com.node);
        float sinn = std::sin(binary_com.node);

        float y1 = dy * cosi;
        float z1 = dy * sini;
        float x1 = dx;
        float bx = cosn * x1 - sinn * y1;
        float by = sinn * x1 + cosn * y1;
        float bz = z1;

        float tx = -std::sin(bin_phase_t);
        float ty = std::cos(bin_phase_t);
        float ty1 = ty * cosi;
        float tz1 = ty * sini;
        float btx = cosn * tx - sinn * ty1;
        float bty = sinn * tx + cosn * ty1;
        float btz = tz1;

        bh_pos[0] = make_float3(com_pos.x + bx, com_pos.y + by, com_pos.z + bz);
        bh_pos[1] = make_float3(com_pos.x - bx, com_pos.y - by, com_pos.z - bz);
        bh_tan[0] = make_float3(com_tan.x + btx, com_tan.y + bty, com_tan.z + btz);
        bh_tan[1] = make_float3(com_tan.x - btx, com_tan.y - bty, com_tan.z - btz);

        for (int i = 0; i < 6; ++i) {
            float3 pos, tan;
            compute_orbit_state(singles[i], t, pos, tan);
            bh_pos[i + 2] = pos;
            bh_tan[i + 2] = tan;
        }

        check_cuda(cudaMemcpy(d_bh_pos, bh_pos.data(), sizeof(float3) * n_bh, cudaMemcpyHostToDevice), "copy bh_pos");
        check_cuda(cudaMemcpy(d_bh_tan, bh_tan.data(), sizeof(float3) * n_bh, cudaMemcpyHostToDevice), "copy bh_tan");

        generate_flow<<<grid, block>>>(
            d_bh_pos, d_bh_tan, d_orient, d_spin_offset,
            d_hot_base, d_hot_r, d_hot_theta, d_hot_out,
            points_per_bh, n_bh, t, hot, global);
        generate_flow<<<grid, block>>>(
            d_bh_pos, d_bh_tan, d_orient, d_spin_offset,
            d_cold_base, d_cold_r, d_cold_theta, d_cold_out,
            points_per_bh, n_bh, t, cold, global);

        check_cuda(cudaGetLastError(), "generate_flow launch");
        check_cuda(cudaDeviceSynchronize(), "generate_flow sync");

        check_cuda(cudaMemcpy(h_hot_out, d_hot_out, out_bytes, cudaMemcpyDeviceToHost), "copy hot out");
        check_cuda(cudaMemcpy(h_cold_out, d_cold_out, out_bytes, cudaMemcpyDeviceToHost), "copy cold out");

        char name[64];
        std::snprintf(name, sizeof(name), "hot_%04d.bin", frame);
        write_frame(hot_dir / name, h_hot_out, out_bytes);
        std::snprintf(name, sizeof(name), "cold_%04d.bin", frame);
        write_frame(cold_dir / name, h_cold_out, out_bytes);

        std::snprintf(name, sizeof(name), "bh_%04d.bin", frame);
        write_frame(bh_dir / name, reinterpret_cast<float *>(bh_pos.data()), sizeof(float3) * n_bh);

        if (frame % 200 == 0) {
            std::printf("Generated frame %d/%d\n", frame, frames);
        }
    }

    cudaFreeHost(h_hot_out);
    cudaFreeHost(h_cold_out);

    cudaFree(d_hot_base);
    cudaFree(d_cold_base);
    cudaFree(d_hot_r);
    cudaFree(d_cold_r);
    cudaFree(d_hot_theta);
    cudaFree(d_cold_theta);
    cudaFree(d_bh_pos);
    cudaFree(d_bh_tan);
    cudaFree(d_orient);
    cudaFree(d_spin_offset);
    cudaFree(d_hot_out);
    cudaFree(d_cold_out);

    std::ofstream meta(base_dir / "meta.json");
    meta << "{\n";
    meta << "  \"n_bh\": " << n_bh << ",\n";
    meta << "  \"points_per_bh\": " << points_per_bh << ",\n";
    meta << "  \"frames\": " << frames << ",\n";
    meta << "  \"dt\": " << dt << "\n";
    meta << "}\n";

    return 0;
}
