#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <cstring>
#include "poisson_fft.cuh"

using half = __half;
using namespace nvcuda;

namespace {
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;
constexpr float GAMMA = 1.4f;
constexpr float RHO_MIN = 1e-6f;
constexpr float P_MIN = 1e-6f;
constexpr float CFL = 0.25f;
constexpr float PI = 3.14159265358979323846f;
}

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do { auto err=(x); if (err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)
#endif

#ifndef CHECK_LAST
#define CHECK_LAST() do { auto err=cudaPeekAtLastError(); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)
#endif

__device__ __forceinline__ float rcp_approx(float x){
    float r;
    asm("rcp.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

__device__ __forceinline__ float safe_rho(float rho){ return fmaxf(rho, RHO_MIN); }

template<bool UseApprox>
__device__ __forceinline__ float inv_rho(float rho){
    float r = safe_rho(rho);
    if constexpr (UseApprox){
        return rcp_approx(r);
    } else {
        return 1.0f / r;
    }
}

template<bool UseApprox>
__device__ __forceinline__ float pressure(float rho, float mx, float my, float bx, float by, float E){
    float invr = inv_rho<UseApprox>(rho);
    float vx = mx * invr;
    float vy = my * invr;
    float r  = safe_rho(rho);
    float kin = 0.5f * r * (vx*vx + vy*vy);
    float mag = 0.5f * (bx*bx + by*by);
    float p = (GAMMA - 1.f) * (E - kin - mag);
    return fmaxf(p, P_MIN);
}

template<bool UseApprox>
__device__ __forceinline__ float fast_magnetosonic(float rho, float p, float bx, float by){
    float invr = inv_rho<UseApprox>(rho);
    float cs2 = fmaxf(GAMMA * p * invr, 1e-12f);
    float b2  = bx*bx + by*by;
    float vA2 = b2 * invr;
    return sqrtf(cs2 + vA2);
}

template<bool UseApprox>
__device__ __forceinline__ void flux_x(const float U[6], float F[6]){
    float rho=U[0], mx=U[1], my=U[2], bx=U[3], by=U[4], E=U[5];
    float invr = inv_rho<UseApprox>(rho);
    float vx = mx * invr, vy = my * invr;
    float p  = pressure<UseApprox>(rho,mx,my,bx,by,E);
    float bt2 = bx*bx + by*by;
    float pt  = p + 0.5f*bt2;
    float vb  = vx*bx + vy*by;
    F[0] = mx;
    F[1] = mx*vx + pt - bx*bx;
    F[2] = mx*vy - bx*by;
    F[3] = 0.0f;
    F[4] = vy*bx - vx*by;
    F[5] = (E + pt)*vx - bx*vb;
}

template<bool UseApprox>
__device__ __forceinline__ void flux_y(const float U[6], float G[6]){
    float rho=U[0], mx=U[1], my=U[2], bx=U[3], by=U[4], E=U[5];
    float invr = inv_rho<UseApprox>(rho);
    float vx = mx * invr, vy = my * invr;
    float p  = pressure<UseApprox>(rho,mx,my,bx,by,E);
    float bt2 = bx*bx + by*by;
    float pt  = p + 0.5f*bt2;
    float vb  = vx*bx + vy*by;
    G[0] = my;
    G[1] = my*vx - by*bx;
    G[2] = my*vy + pt - by*by;
    G[3] = vx*by - vy*bx;
    G[4] = 0.0f;
    G[5] = (E + pt)*vy - by*vb;
}

template<bool UseApprox>
__device__ __forceinline__ void rusanov_flux(
    const float UL[6], const float UR[6], int dirXY, float outF[6])
{
    float FL[6], FR[6];
    if (dirXY==0){ flux_x<UseApprox>(UL, FL); flux_x<UseApprox>(UR, FR); }
    else         { flux_y<UseApprox>(UL, FL); flux_y<UseApprox>(UR, FR); }

    float invrL = inv_rho<UseApprox>(UL[0]);
    float invrR = inv_rho<UseApprox>(UR[0]);
    float vxL=UL[1]*invrL, vyL=UL[2]*invrL, vxR=UR[1]*invrR, vyR=UR[2]*invrR;
    float pL = pressure<UseApprox>(UL[0],UL[1],UL[2],UL[3],UL[4],UL[5]);
    float pR = pressure<UseApprox>(UR[0],UR[1],UR[2],UR[3],UR[4],UR[5]);

    float cfL = fast_magnetosonic<UseApprox>(UL[0],pL,UL[3],UL[4]);
    float cfR = fast_magnetosonic<UseApprox>(UR[0],pR,UR[3],UR[4]);

    float vnL = (dirXY==0) ? vxL : vyL;
    float vnR = (dirXY==0) ? vxR : vyR;
    float smax = fmaxf(fabsf(vnL)+cfL, fabsf(vnR)+cfR);

    #pragma unroll
    for (int m=0;m<6;m++)
        outF[m] = 0.5f*(FL[m] + FR[m]) - 0.5f*smax*(UR[m] - UL[m]);
}

__device__ __forceinline__ int wrap(int i, int n){ i%=n; if(i<0) i+=n; return i; }

__device__ __forceinline__ void load_U_half(const half* U, int idx, int stride, float out[6]){
    out[0] = __half2float(U[0*stride + idx]);
    out[1] = __half2float(U[1*stride + idx]);
    out[2] = __half2float(U[2*stride + idx]);
    out[3] = __half2float(U[3*stride + idx]);
    out[4] = __half2float(U[4*stride + idx]);
    out[5] = __half2float(U[5*stride + idx]);
}

__device__ __forceinline__ void store_U_half(half* U, int idx, int stride, const float in[6]){
    float Un[6];
    #pragma unroll
    for (int c=0;c<6;c++) Un[c]=in[c];
    Un[0] = fmaxf(Un[0], RHO_MIN);
    float invr = 1.0f / safe_rho(Un[0]);
    float vx=Un[1]*invr, vy=Un[2]*invr;
    float kin = 0.5f*Un[0]*(vx*vx + vy*vy);
    float mag = 0.5f*(Un[3]*Un[3] + Un[4]*Un[4]);
    float Emin = kin + mag + P_MIN/(GAMMA-1.f);
    if (Un[5] < Emin) Un[5] = Emin;

    U[0*stride + idx] = __float2half_rn(Un[0]);
    U[1*stride + idx] = __float2half_rn(Un[1]);
    U[2*stride + idx] = __float2half_rn(Un[2]);
    U[3*stride + idx] = __float2half_rn(Un[3]);
    U[4*stride + idx] = __float2half_rn(Un[4]);
    U[5*stride + idx] = __float2half_rn(Un[5]);
}

template<bool UseApprox>
__global__ void step_mhd_rusanov_half(
    const half* __restrict__ Uin,
    half* __restrict__ Uout,
    int NX, int NY, float dx, float dy, float dt)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=NX || iy>=NY) return;
    int stride = NX*NY;
    int idx = iy*NX + ix;

    int im1 = wrap(ix-1,NX), ip1 = wrap(ix+1,NX);
    int jm1 = wrap(iy-1,NY), jp1 = wrap(iy+1,NY);

    float Uc[6], Ul[6], Ur[6], Ud[6], Uu[6];
    load_U_half(Uin, idx, stride, Uc);
    load_U_half(Uin, iy*NX + im1, stride, Ul);
    load_U_half(Uin, iy*NX + ip1, stride, Ur);
    load_U_half(Uin, jm1*NX + ix, stride, Ud);
    load_U_half(Uin, jp1*NX + ix, stride, Uu);

    float FxL[6], FxR[6], GyD[6], GyU[6];
    rusanov_flux<UseApprox>(Ul, Uc, 0, FxL);
    rusanov_flux<UseApprox>(Uc, Ur, 0, FxR);
    rusanov_flux<UseApprox>(Ud, Uc, 1, GyD);
    rusanov_flux<UseApprox>(Uc, Uu, 1, GyU);

    float Un[6];
    float dtdx = dt / dx;
    float dtdy = dt / dy;
    #pragma unroll
    for (int c=0;c<6;c++){
        Un[c] = Uc[c] - dtdx*(FxR[c]-FxL[c]) - dtdy*(GyU[c]-GyD[c]);
    }
    store_U_half(Uout, idx, stride, Un);
}

__device__ unsigned int d_max_speed_bits;

__global__ void reset_maxspeed(){ d_max_speed_bits = 0u; }

__device__ __forceinline__ void atomicMaxFloatPos(float v){
    atomicMax(&d_max_speed_bits, __float_as_uint(v));
}

template<bool UseApprox>
__global__ void kernel_maxspeed_half(const half* __restrict__ U, int NX, int NY){
    int ix=blockIdx.x*blockDim.x + threadIdx.x;
    int iy=blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=NX || iy>=NY) return;
    int stride = NX*NY;
    int idx=iy*NX+ix;

    float rho=__half2float(U[0*stride + idx]);
    float mx =__half2float(U[1*stride + idx]);
    float my =__half2float(U[2*stride + idx]);
    float bx =__half2float(U[3*stride + idx]);
    float by =__half2float(U[4*stride + idx]);
    float E  =__half2float(U[5*stride + idx]);

    float invr = inv_rho<UseApprox>(rho);
    float vx = mx * invr, vy = my * invr;
    float p  = pressure<UseApprox>(rho,mx,my,bx,by,E);
    float cf = fast_magnetosonic<UseApprox>(rho,p,bx,by);

    float smax = fmaxf(fabsf(vx)+cf, fabsf(vy)+cf);
    atomicMaxFloatPos(fmaxf(smax,0.f));
}

static inline float uint_as_float_host(uint32_t u){
    float f; std::memcpy(&f,&u,sizeof(float)); return f;
}

template<bool UseApprox>
float compute_dt_gpu_half(const half* dU, int NX, int NY, float dx, float dy, float dt_scale){
    dim3 block(16,16), grid((NX+block.x-1)/block.x,(NY+block.y-1)/block.y);
    reset_maxspeed<<<1,1>>>();
    kernel_maxspeed_half<UseApprox><<<grid,block>>>(dU,NX,NY);
    CHECK_CUDA(cudaDeviceSynchronize());
    uint32_t hbits=0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&hbits,d_max_speed_bits,sizeof(uint32_t)));
    float smax = uint_as_float_host(hbits);
    if (!std::isfinite(smax) || smax<=0.f) smax = 1.0f;
    float dt = CFL * fminf(dx,dy) / smax;
    return dt * dt_scale;
}

__global__ void extract_rho_half_to_float(const half* rho_half, float* rho_float, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) rho_float[i] = __half2float(rho_half[i]);
}

__global__ void apply_grad_phi_half(half* U, const float* phi,
                                    int NX, int NY, float dx, float dy, float dt)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix>=NX || iy>=NY) return;
    int xm1 = (ix+NX-1)%NX, xp1 = (ix+1)%NX;
    int ym1 = (iy+NY-1)%NY, yp1 = (iy+1)%NY;
    float dphix = (phi[iy*NX + xp1] - phi[iy*NX + xm1]) / (2.0f*dx);
    float dphiy = (phi[yp1*NX + ix] - phi[ym1*NX + ix]) / (2.0f*dy);
    int stride = NX*NY;
    int idx=iy*NX+ix;
    float mx = __half2float(U[1*stride + idx]);
    float my = __half2float(U[2*stride + idx]);
    mx -= dt * dphix;
    my -= dt * dphiy;
    U[1*stride + idx] = __float2half_rn(mx);
    U[2*stride + idx] = __float2half_rn(my);
}

__global__ void reduce_sum(const float* data, double* out, int N){
    __shared__ double sh[256];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    double v = (i < N) ? static_cast<double>(data[i]) : 0.0;
    sh[threadIdx.x] = v;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1){
        if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(out, sh[0]);
}

__global__ void wmma_blur_rho(half* rho, const half* A, int NX, int NY, float mix){
    int tile_x = blockIdx.x;
    int tile_y = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = tile_x * WMMA_N + tx;
    int y = tile_y * WMMA_M + ty;

    __shared__ float sh_C[WMMA_M * WMMA_N];
    int lane = ty * blockDim.x + tx;

    if (lane < 32){
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> a;
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> b;
        wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> c;
        wmma::load_matrix_sync(a, A, WMMA_K);
        const half* tile_ptr = rho + (tile_y * WMMA_M * NX + tile_x * WMMA_N);
        wmma::load_matrix_sync(b, tile_ptr, NX);
        wmma::fill_fragment(c, 0.0f);
        wmma::mma_sync(c, a, b, c);
        wmma::store_matrix_sync(sh_C, c, WMMA_N, wmma::mem_row_major);
    }
    __syncthreads();

    if (x < NX && y < NY){
        int idx = y * NX + x;
        float rho_old = __half2float(rho[idx]);
        float rho_smooth = sh_C[ty * WMMA_N + tx];
        float rho_new = (1.0f - mix) * rho_old + mix * rho_smooth;
        rho_new = fmaxf(rho_new, RHO_MIN);
        rho[idx] = __float2half_rn(rho_new);
    }
}

static void init_orszag_tang(std::vector<float>& hU, int NX, int NY){
    auto at = [&](int c,int x,int y)->float& { return hU[c*NX*NY + y*NX + x]; };
    for (int y=0;y<NY;y++){
        for (int x=0;x<NX;x++){
            float X = (x + 0.5f) / float(NX);
            float Y = (y + 0.5f) / float(NY);
            float rho=1.0f;
            float vx =-sinf(2.f*PI*Y);
            float vy = sinf(2.f*PI*X);
            float bx =-sinf(2.f*PI*Y);
            float by = sinf(4.f*PI*X);
            float p  = 1.0f;
            float mx=rho*vx, my=rho*vy;
            float kin=0.5f*rho*(vx*vx+vy*vy);
            float mag=0.5f*(bx*bx+by*by);
            float E = p/(GAMMA-1.f) + kin + mag;
            at(0,x,y)=rho; at(1,x,y)=mx; at(2,x,y)=my;
            at(3,x,y)=bx;  at(4,x,y)=by; at(5,x,y)=E;
        }
    }
}

static void dump_field_dir(const char* dir, const std::vector<float>& data,
                           const char* tag, int step){
    std::filesystem::create_directories(dir);
    char name[256];
    std::snprintf(name, sizeof(name), "%s/%s_%04d.bin", dir, tag, step);
    FILE* fp = std::fopen(name, "wb");
    if (!fp){
        std::fprintf(stderr, "Failed to write %s\n", name);
        return;
    }
    std::fwrite(data.data(), sizeof(float), data.size(), fp);
    std::fclose(fp);
}

template<bool UseApprox>
void run_sim(const char* tag,
             const std::vector<half>& hU_half,
             half* dU, half* dV, float* d_rho_f, float* d_phi, double* d_sum,
             Poisson2D& poisson, const half* d_A,
             int NX, int NY, int STEPS, float dt_scale, float phi_scale, float wmma_mix,
             int dump_every)
{
    size_t cells = static_cast<size_t>(NX) * NY;
    size_t bytes_h = 6 * cells * sizeof(half);
    size_t bytes_f = cells * sizeof(float);
    CHECK_CUDA(cudaMemcpy(dU, hU_half.data(), bytes_h, cudaMemcpyHostToDevice));

    dim3 block(16,16), grid((NX+block.x-1)/block.x,(NY+block.y-1)/block.y);
    dim3 block_tc(WMMA_N, WMMA_M);
    dim3 grid_tc(NX / WMMA_N, NY / WMMA_M);
    int tb = 256;
    int gb = (int)((cells + tb - 1) / tb);

    std::vector<float> h_rho(cells), h_phi(cells);
    float Lx=1.0f, Ly=1.0f;
    float dx=Lx/NX, dy=Ly/NY;

    for (int s=0; s<STEPS; ++s){
        float dt = compute_dt_gpu_half<UseApprox>(dU, NX, NY, dx, dy, dt_scale);
        step_mhd_rusanov_half<UseApprox><<<grid,block>>>(dU,dV,NX,NY,dx,dy,dt);
        CHECK_LAST();
        CHECK_CUDA(cudaDeviceSynchronize());
        std::swap(dU,dV);

        extract_rho_half_to_float<<<gb,tb>>>(dU, d_rho_f, (int)cells);
        CHECK_LAST();
        CHECK_CUDA(cudaMemset(d_sum, 0, sizeof(double)));
        reduce_sum<<<gb,tb>>>(d_rho_f, d_sum, (int)cells);
        CHECK_LAST();
        CHECK_CUDA(cudaDeviceSynchronize());
        double h_sum = 0.0;
        CHECK_CUDA(cudaMemcpy(&h_sum, d_sum, sizeof(double), cudaMemcpyDeviceToHost));
        float mean_rho = static_cast<float>(h_sum / double(cells));
        poisson.solve(d_rho_f, d_phi, mean_rho, dx, dy);

        apply_grad_phi_half<<<grid,block>>>(dU, d_phi, NX, NY, dx, dy, dt * phi_scale);
        CHECK_LAST();
        CHECK_CUDA(cudaDeviceSynchronize());

        if (wmma_mix > 0.0f){
            wmma_blur_rho<<<grid_tc, block_tc>>>(dU, d_A, NX, NY, wmma_mix);
            CHECK_LAST();
            CHECK_CUDA(cudaDeviceSynchronize());
        }

        if (dump_every > 0 && (s % dump_every) == 0){
            CHECK_CUDA(cudaMemcpy(h_rho.data(), d_rho_f, bytes_f, cudaMemcpyDeviceToHost));
            CHECK_CUDA(cudaMemcpy(h_phi.data(), d_phi, bytes_f, cudaMemcpyDeviceToHost));
            dump_field_dir(tag, h_rho, "rho", s);
            dump_field_dir(tag, h_phi, "phi", s);
        }
    }
}

int main(int argc, char** argv){
    int NX = (argc > 1) ? std::atoi(argv[1]) : 128;
    int NY = (argc > 2) ? std::atoi(argv[2]) : 128;
    int STEPS = (argc > 3) ? std::atoi(argv[3]) : 120;
    float dt_scale = (argc > 4) ? std::atof(argv[4]) : 1.0f;
    float phi_scale = (argc > 5) ? std::atof(argv[5]) : 0.08f;
    float wmma_mix = (argc > 6) ? std::atof(argv[6]) : 0.05f;
    int dump_every = (argc > 7) ? std::atoi(argv[7]) : 1;

    if (NX % WMMA_N != 0 || NY % WMMA_M != 0){
        std::fprintf(stderr, "NX and NY must be multiples of %d\n", WMMA_N);
        return 1;
    }

    size_t cells = static_cast<size_t>(NX) * NY;
    size_t bytes_h = 6 * cells * sizeof(half);
    size_t bytes_f = cells * sizeof(float);

    std::vector<float> hU(6*cells);
    init_orszag_tang(hU, NX, NY);
    std::vector<half> hU_half(6*cells);
    for (size_t i=0;i<6*cells;i++) hU_half[i] = __float2half_rn(hU[i]);

    half *dU=nullptr, *dV=nullptr;
    float *d_rho_f=nullptr, *d_phi=nullptr;
    double *d_sum=nullptr;
    CHECK_CUDA(cudaMalloc(&dU, bytes_h));
    CHECK_CUDA(cudaMalloc(&dV, bytes_h));
    CHECK_CUDA(cudaMalloc(&d_rho_f, bytes_f));
    CHECK_CUDA(cudaMalloc(&d_phi, bytes_f));
    CHECK_CUDA(cudaMalloc(&d_sum, sizeof(double)));

    // WMMA smoothing matrix A.
    std::vector<half> h_A(WMMA_M * WMMA_K);
    float alpha = 0.12f;
    for (int i = 0; i < WMMA_M; ++i){
        for (int j = 0; j < WMMA_K; ++j){
            float v = 0.0f;
            if (i == j) v = 1.0f - 2.0f * alpha;
            if (j == i - 1 || j == i + 1) v = alpha;
            if (i == 0 && j == 0) v = 1.0f - alpha;
            if (i == WMMA_M - 1 && j == WMMA_M - 1) v = 1.0f - alpha;
            h_A[i * WMMA_K + j] = __float2half_rn(v);
        }
    }
    half* d_A = nullptr;
    CHECK_CUDA(cudaMalloc(&d_A, h_A.size() * sizeof(half)));
    CHECK_CUDA(cudaMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), cudaMemcpyHostToDevice));

    Poisson2D poisson;
    poisson.init(NX, NY);

    run_sim<false>("frames_exact", hU_half, dU, dV, d_rho_f, d_phi, d_sum,
                   poisson, d_A, NX, NY, STEPS, dt_scale, phi_scale, wmma_mix,
                   dump_every);
    run_sim<true>("frames_approx", hU_half, dU, dV, d_rho_f, d_phi, d_sum,
                  poisson, d_A, NX, NY, STEPS, dt_scale, phi_scale, wmma_mix,
                  dump_every);

    poisson.destroy();
    cudaFree(d_A);
    cudaFree(dU);
    cudaFree(dV);
    cudaFree(d_rho_f);
    cudaFree(d_phi);
    cudaFree(d_sum);
    return 0;
}
