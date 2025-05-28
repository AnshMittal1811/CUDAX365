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

__global__ void flux_baseline(const float *rho, const float *phi, float *out, int nx, int ny, float dt) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) {
        return;
    }
    int idx = y * nx + x;
    int xm = (x > 0) ? x - 1 : x;
    int xp = (x + 1 < nx) ? x + 1 : x;
    int ym = (y > 0) ? y - 1 : y;
    int yp = (y + 1 < ny) ? y + 1 : y;

    float rho_c = rho[idx];
    float phi_x = phi[y * nx + xp] - phi[y * nx + xm];
    float phi_y = phi[yp * nx + x] - phi[ym * nx + x];

    float inv_rho = 1.0f / (rho_c + 1e-3f);
    float flux = (phi_x + phi_y) * inv_rho;
    out[idx] = rho_c - dt * flux;
}

__global__ void flux_optimized(const float *rho, const float *phi, float *out, int nx, int ny, float dt) {
    __shared__ float tile_rho[18 * 18];
    __shared__ float tile_phi[18 * 18];

    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * 16 + tx;
    int y = blockIdx.y * 16 + ty;

    int tile_x = blockIdx.x * 16 + tx - 1;
    int tile_y = blockIdx.y * 16 + ty - 1;
    int clamped_x = min(max(tile_x, 0), nx - 1);
    int clamped_y = min(max(tile_y, 0), ny - 1);
    int tile_idx = ty * 18 + tx;
    int global_idx = clamped_y * nx + clamped_x;

    tile_rho[tile_idx] = rho[global_idx];
    tile_phi[tile_idx] = phi[global_idx];
    __syncthreads();

    if (tx >= 1 && tx < 17 && ty >= 1 && ty < 17 && x < nx && y < ny) {
        int center = ty * 18 + tx;
        float rho_c = tile_rho[center];
        float phi_x = tile_phi[center + 1] - tile_phi[center - 1];
        float phi_y = tile_phi[center + 18] - tile_phi[center - 18];
        float inv_rho = __frcp_rn(rho_c + 1e-3f);
        float flux = __fmaf_rn(phi_x + phi_y, inv_rho, 0.0f);
        out[y * nx + x] = rho_c - dt * flux;
    }
}

static float time_kernel(dim3 grid, dim3 block, const float *rho, const float *phi, float *out,
                         int nx, int ny, float dt, bool optimized) {
    cudaEvent_t start, stop;
    check(cudaEventCreate(&start), "event create start");
    check(cudaEventCreate(&stop), "event create stop");
    check(cudaEventRecord(start), "record start");
    if (optimized) {
        flux_optimized<<<grid, block>>>(rho, phi, out, nx, ny, dt);
    } else {
        flux_baseline<<<grid, block>>>(rho, phi, out, nx, ny, dt);
    }
    check(cudaEventRecord(stop), "record stop");
    check(cudaEventSynchronize(stop), "sync stop");
    float ms = 0.0f;
    check(cudaEventElapsedTime(&ms, start, stop), "elapsed");
    check(cudaEventDestroy(start), "destroy start");
    check(cudaEventDestroy(stop), "destroy stop");
    return ms;
}

int main(int argc, char **argv) {
    int nx = (argc > 1) ? std::atoi(argv[1]) : 512;
    int ny = (argc > 2) ? std::atoi(argv[2]) : 512;
    float dt = (argc > 3) ? std::atof(argv[3]) : 0.1f;

    if (nx <= 0 || ny <= 0) {
        std::fprintf(stderr, "invalid grid size\n");
        return 1;
    }

    size_t total = static_cast<size_t>(nx) * ny;
    std::vector<float> h_rho(total, 1.0f);
    std::vector<float> h_phi(total, 0.5f);

    float *d_rho = nullptr;
    float *d_phi = nullptr;
    float *d_out = nullptr;
    check(cudaMalloc(&d_rho, total * sizeof(float)), "malloc d_rho");
    check(cudaMalloc(&d_phi, total * sizeof(float)), "malloc d_phi");
    check(cudaMalloc(&d_out, total * sizeof(float)), "malloc d_out");

    check(cudaMemcpy(d_rho, h_rho.data(), total * sizeof(float), cudaMemcpyHostToDevice), "H2D rho");
    check(cudaMemcpy(d_phi, h_phi.data(), total * sizeof(float), cudaMemcpyHostToDevice), "H2D phi");

    dim3 block(16, 16);
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y);

    float ms_base = time_kernel(grid, block, d_rho, d_phi, d_out, nx, ny, dt, false);
    float ms_opt = time_kernel(grid, dim3(18, 18), d_rho, d_phi, d_out, nx, ny, dt, true);

    std::printf("baseline_ms=%.4f optimized_ms=%.4f speedup=%.2fx\n",
                ms_base, ms_opt, ms_base / ms_opt);

    check(cudaFree(d_rho), "free d_rho");
    check(cudaFree(d_phi), "free d_phi");
    check(cudaFree(d_out), "free d_out");
    return 0;
}
