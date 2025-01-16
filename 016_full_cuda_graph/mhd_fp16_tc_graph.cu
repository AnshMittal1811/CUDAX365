#include <cuda_runtime.h>
#include <mma.h>
#include <cuda_fp16.h>
#include <cufft.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <algorithm>
#include <filesystem>
#include <cstring>

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

#ifndef CHECK_CUFFT
#define CHECK_CUFFT(x) do { auto st=(x); if(st!=CUFFT_SUCCESS){ \
  fprintf(stderr,"cuFFT error %s:%d: code %d\n", __FILE__,__LINE__,(int)st); exit(1);} } while(0)
#endif

#ifndef CHECK_LAST
#define CHECK_LAST() do { auto err=cudaPeekAtLastError(); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)
#endif

__device__ __forceinline__ float safe_rho(float rho){ return fmaxf(rho, RHO_MIN); }

__device__ __forceinline__ float pressure(float rho, float mx, float my, float bx, float by, float E){
    float r  = safe_rho(rho);
    float vx = mx / r, vy = my / r;
    float kin = 0.5f * r * (vx*vx + vy*vy);
    float mag = 0.5f * (bx*bx + by*by);
    float p = (GAMMA - 1.f) * (E - kin - mag);
    return fmaxf(p, P_MIN);
}

__device__ __forceinline__ float fast_magnetosonic(float rho, float p, float bx, float by){
    float r   = safe_rho(rho);
    float cs2 = fmaxf(GAMMA * p / r, 1e-12f);
    float b2  = bx*bx + by*by;
    float vA2 = b2 / r;
    return sqrtf(cs2 + vA2);
}

__device__ __forceinline__ void flux_x(const float U[6], float F[6]){
    float rho=U[0], mx=U[1], my=U[2], bx=U[3], by=U[4], E=U[5];
    float r  = safe_rho(rho);
    float vx = mx/r, vy = my/r;
    float p  = pressure(rho,mx,my,bx,by,E);
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

__device__ __forceinline__ void flux_y(const float U[6], float G[6]){
    float rho=U[0], mx=U[1], my=U[2], bx=U[3], by=U[4], E=U[5];
    float r  = safe_rho(rho);
    float vx = mx/r, vy = my/r;
    float p  = pressure(rho,mx,my,bx,by,E);
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

__device__ __forceinline__ void rusanov_flux(
    const float UL[6], const float UR[6], int dirXY, float outF[6])
{
    float FL[6], FR[6];
    if (dirXY==0){ flux_x(UL, FL); flux_x(UR, FR); }
    else         { flux_y(UL, FL); flux_y(UR, FR); }

    float rL = safe_rho(UL[0]), rR = safe_rho(UR[0]);
    float vxL=UL[1]/rL, vyL=UL[2]/rL, vxR=UR[1]/rR, vyR=UR[2]/rR;
    float pL = pressure(UL[0],UL[1],UL[2],UL[3],UL[4],UL[5]);
    float pR = pressure(UR[0],UR[1],UR[2],UR[3],UR[4],UR[5]);

    float cfL = fast_magnetosonic(UL[0],pL,UL[3],UL[4]);
    float cfR = fast_magnetosonic(UR[0],pR,UR[3],UR[4]);

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
    float r = safe_rho(Un[0]);
    float vx=Un[1]/r, vy=Un[2]/r;
    float kin = 0.5f*r*(vx*vx + vy*vy);
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

__global__ void step_mhd_rusanov_half_graph(
    const half* __restrict__ Uin,
    half* __restrict__ Uout,
    const float* __restrict__ d_dt,
    int NX, int NY, float dx, float dy)
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
    rusanov_flux(Ul, Uc, 0, FxL);
    rusanov_flux(Uc, Ur, 0, FxR);
    rusanov_flux(Ud, Uc, 1, GyD);
    rusanov_flux(Uc, Uu, 1, GyU);

    float dt = d_dt[0];
    float dtdx = dt / dx;
    float dtdy = dt / dy;
    float Un[6];
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

    float r  = safe_rho(rho);
    float vx = mx/r, vy=my/r;
    float p  = pressure(rho,mx,my,bx,by,E);
    float cf = fast_magnetosonic(rho,p,bx,by);

    float smax = fmaxf(fabsf(vx)+cf, fabsf(vy)+cf);
    atomicMaxFloatPos(fmaxf(smax,0.f));
}

__global__ void compute_dt_kernel(float* d_dt, float dx, float dy, float dt_scale){
    float smax = __uint_as_float(d_max_speed_bits);
    if (!isfinite(smax) || smax <= 0.0f) smax = 1.0f;
    float dt = CFL * fminf(dx, dy) / smax;
    d_dt[0] = dt * dt_scale;
}

__global__ void extract_rho_half_to_float(const half* rho_half, float* rho_float, int N){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < N) rho_float[i] = __half2float(rho_half[i]);
}

__global__ void apply_grad_phi_half_graph(half* U, const float* phi, const float* d_dt,
                                          int NX, int NY, float dx, float dy, float phi_scale)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix>=NX || iy>=NY) return;
    int xm1 = (ix+NX-1)%NX, xp1 = (ix+1)%NX;
    int ym1 = (iy+NY-1)%NY, yp1 = (iy+1)%NY;
    float dphix = (phi[iy*NX + xp1] - phi[iy*NX + xm1]) / (2.0f*dx);
    float dphiy = (phi[yp1*NX + ix] - phi[ym1*NX + ix]) / (2.0f*dy);
    float dt = d_dt[0] * phi_scale;
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

__global__ void compute_mean_kernel(const double* sum, float* mean, int N){
    mean[0] = static_cast<float>(sum[0] / static_cast<double>(N));
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

__global__ void store_frame(const float* rho, const float* phi,
                            float* rho_frames, float* phi_frames,
                            int N, int frame_idx)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        int base = frame_idx * N + i;
        rho_frames[base] = rho[i];
        phi_frames[base] = phi[i];
    }
}

__global__ void fill_k2(float* __restrict__ k2, int NX, int NY){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix>=NX || iy>=NY) return;
    int nx = (ix<=NX/2)? ix : ix-NX;
    int ny = (iy<=NY/2)? iy : iy-NY;
    float kx = 2.0f*PI*nx;
    float ky = 2.0f*PI*ny;
    k2[iy*NX + ix] = kx*kx + ky*ky;
}

__global__ void apply_poisson_scaling(cufftComplex* __restrict__ F,
                                      const float* __restrict__ k2,
                                      int NX, int NY)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix>=NX || iy>=NY) return;
    int idx = iy*NX + ix;
    float kk = k2[idx];
    cufftComplex v = F[idx];
    if (ix==0 && iy==0){ F[idx].x = 0.0f; F[idx].y = 0.0f; return; }
    float s = (kk>0.f)? (-1.0f/kk) : 0.0f;
    F[idx].x = v.x * s;
    F[idx].y = v.y * s;
}

__global__ void subtract_mean_device(float* __restrict__ f, int N, const float* mean){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N) f[i] -= mean[0];
}

__global__ void real_to_complex(cufftComplex* dst, const float* src, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) dst[i]=make_cuFloatComplex(src[i],0.0f);
}

__global__ void complex_to_real(float* out, const cufftComplex* in, int N, float invN){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) out[i] = in[i].x * invN;
}

struct Poisson2DGraph {
    int NX{}, NY{};
    cufftHandle planF{}, planI{};
    float *d_k2{};
    cufftComplex *d_tmp{};
    bool ready{false};

    void init(int NX_, int NY_, cudaStream_t stream){
        NX=NX_; NY=NY_;
        CHECK_CUDA(cudaMalloc(&d_k2,  NX*NY*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_tmp, NX*NY*sizeof(cufftComplex)));
        dim3 b(16,16), g((NX+15)/16,(NY+15)/16);
        fill_k2<<<g,b,0,stream>>>(d_k2, NX, NY);
        CHECK_CUDA(cudaStreamSynchronize(stream));
        CHECK_CUFFT(cufftPlan2d(&planF, NY, NX, CUFFT_C2C));
        CHECK_CUFFT(cufftPlan2d(&planI, NY, NX, CUFFT_C2C));
        CHECK_CUFFT(cufftSetStream(planF, stream));
        CHECK_CUFFT(cufftSetStream(planI, stream));
        ready=true;
    }
    void destroy(){
        if(!ready) return;
        cufftDestroy(planF); cufftDestroy(planI);
        cudaFree(d_k2); cudaFree(d_tmp);
        ready=false;
    }

    void solve(const float* rhs_real, float* phi_real, const float* mean_ptr,
               cudaStream_t stream)
    {
        size_t N = (size_t)NX*NY;
        CHECK_CUDA(cudaMemcpyAsync(phi_real, rhs_real, N*sizeof(float),
                                   cudaMemcpyDeviceToDevice, stream));
        int tb=256, gb=(int)((N+tb-1)/tb);
        subtract_mean_device<<<gb,tb,0,stream>>>(phi_real, (int)N, mean_ptr);
        real_to_complex<<<gb,tb,0,stream>>>(d_tmp, phi_real, (int)N);
        CHECK_CUFFT(cufftExecC2C(planF, d_tmp, d_tmp, CUFFT_FORWARD));
        dim3 b2(16,16), g2((NX+15)/16,(NY+15)/16);
        apply_poisson_scaling<<<g2,b2,0,stream>>>(d_tmp, d_k2, NX, NY);
        CHECK_CUFFT(cufftExecC2C(planI, d_tmp, d_tmp, CUFFT_INVERSE));
        float invN = 1.0f / float(NX*NY);
        complex_to_real<<<gb,tb,0,stream>>>(phi_real, d_tmp, (int)N, invN);
    }
};

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

static void dump_frames(const std::vector<float>& rho_frames,
                        const std::vector<float>& phi_frames,
                        int NX, int NY,
                        const std::vector<int>& steps)
{
    std::filesystem::create_directories("frames");
    size_t cells = static_cast<size_t>(NX) * NY;
    for (size_t i=0;i<steps.size(); ++i){
        int step = steps[i];
        const float* rho = rho_frames.data() + i * cells;
        const float* phi = phi_frames.data() + i * cells;
        char name[256];
        std::snprintf(name, sizeof(name), "frames/rho_%04d.bin", step);
        FILE* fp = std::fopen(name, "wb");
        if (fp){ std::fwrite(rho, sizeof(float), cells, fp); std::fclose(fp); }
        std::snprintf(name, sizeof(name), "frames/phi_%04d.bin", step);
        fp = std::fopen(name, "wb");
        if (fp){ std::fwrite(phi, sizeof(float), cells, fp); std::fclose(fp); }
    }
}

int main(int argc, char** argv){
    int NX = (argc > 1) ? std::atoi(argv[1]) : 128;
    int NY = (argc > 2) ? std::atoi(argv[2]) : 128;
    int STEPS = (argc > 3) ? std::atoi(argv[3]) : 1000;
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
    float *d_mean=nullptr, *d_dt=nullptr;
    CHECK_CUDA(cudaMalloc(&dU, bytes_h));
    CHECK_CUDA(cudaMalloc(&dV, bytes_h));
    CHECK_CUDA(cudaMalloc(&d_rho_f, bytes_f));
    CHECK_CUDA(cudaMalloc(&d_phi, bytes_f));
    CHECK_CUDA(cudaMalloc(&d_sum, sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_mean, sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dt, sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dU, hU_half.data(), bytes_h, cudaMemcpyHostToDevice));

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

    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));

    Poisson2DGraph poisson;
    poisson.init(NX, NY, stream);

    dim3 block(16,16), grid((NX+block.x-1)/block.x,(NY+block.y-1)/block.y);
    dim3 block_tc(WMMA_N, WMMA_M);
    dim3 grid_tc(NX / WMMA_N, NY / WMMA_M);
    int tb = 256;
    int gb = (int)((cells + tb - 1) / tb);

    int frame_count = 0;
    std::vector<int> frame_steps;
    if (dump_every > 0){
        frame_count = (STEPS + dump_every - 1) / dump_every;
        frame_steps.reserve(frame_count);
        for (int s=0; s<STEPS; ++s){
            if ((s % dump_every) == 0) frame_steps.push_back(s);
        }
    }
    float *d_rho_frames=nullptr, *d_phi_frames=nullptr;
    if (frame_count > 0){
        CHECK_CUDA(cudaMalloc(&d_rho_frames, (size_t)frame_count * cells * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_phi_frames, (size_t)frame_count * cells * sizeof(float)));
    }

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    CHECK_CUDA(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
    half* cur = dU;
    half* nxt = dV;
    int frame_idx = 0;
    for (int s=0; s<STEPS; ++s){
        reset_maxspeed<<<1,1,0,stream>>>();
        kernel_maxspeed_half<<<grid,block,0,stream>>>(cur, NX, NY);
        compute_dt_kernel<<<1,1,0,stream>>>(d_dt, 1.0f/float(NX), 1.0f/float(NY), dt_scale);
        step_mhd_rusanov_half_graph<<<grid,block,0,stream>>>(cur, nxt, d_dt, NX, NY,
                                                            1.0f/float(NX), 1.0f/float(NY));
        std::swap(cur, nxt);

        extract_rho_half_to_float<<<gb,tb,0,stream>>>(cur, d_rho_f, (int)cells);
        CHECK_CUDA(cudaMemsetAsync(d_sum, 0, sizeof(double), stream));
        reduce_sum<<<gb,tb,0,stream>>>(d_rho_f, d_sum, (int)cells);
        compute_mean_kernel<<<1,1,0,stream>>>(d_sum, d_mean, (int)cells);
        poisson.solve(d_rho_f, d_phi, d_mean, stream);
        apply_grad_phi_half_graph<<<grid,block,0,stream>>>(cur, d_phi, d_dt,
                                                          NX, NY,
                                                          1.0f/float(NX),
                                                          1.0f/float(NY),
                                                          phi_scale);
        if (wmma_mix > 0.0f){
            wmma_blur_rho<<<grid_tc, block_tc, 0, stream>>>(cur, d_A, NX, NY, wmma_mix);
        }
        if (frame_count > 0 && (s % dump_every) == 0){
            store_frame<<<gb,tb,0,stream>>>(d_rho_f, d_phi,
                                            d_rho_frames, d_phi_frames,
                                            (int)cells, frame_idx);
            frame_idx++;
        }
    }
    CHECK_CUDA(cudaStreamEndCapture(stream, &graph));
    CHECK_CUDA(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));

    CHECK_CUDA(cudaGraphLaunch(graph_exec, stream));
    CHECK_CUDA(cudaStreamSynchronize(stream));

    if (frame_count > 0){
        std::vector<float> h_rho_frames((size_t)frame_count * cells);
        std::vector<float> h_phi_frames((size_t)frame_count * cells);
        CHECK_CUDA(cudaMemcpy(h_rho_frames.data(), d_rho_frames,
                              h_rho_frames.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_phi_frames.data(), d_phi_frames,
                              h_phi_frames.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
        dump_frames(h_rho_frames, h_phi_frames, NX, NY, frame_steps);
    }

    poisson.destroy();
    cudaFree(d_rho_frames);
    cudaFree(d_phi_frames);
    cudaGraphDestroy(graph);
    cudaGraphExecDestroy(graph_exec);
    cudaFree(d_A);
    cudaFree(dU);
    cudaFree(dV);
    cudaFree(d_rho_f);
    cudaFree(d_phi);
    cudaFree(d_sum);
    cudaFree(d_mean);
    cudaFree(d_dt);
    cudaStreamDestroy(stream);
    return 0;
}
