#pragma once
#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <vector>
#include <cmath>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do{auto err=(x); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n",__FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} }while(0)
#endif

#ifndef CHECK_CUFFT
#define CHECK_CUFFT(x) do{auto st=(x); if(st!=CUFFT_SUCCESS){ \
  fprintf(stderr,"cuFFT error %s:%d: code %d\n",__FILE__,__LINE__,(int)st); exit(1);} }while(0)
#endif

// --- make k-grid (single-precision) for periodic box [0,1]x[0,1] ---
__global__ void fill_k2(float* __restrict__ k2, int NX, int NY){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix>=NX || iy>=NY) return;

    // frequencies in radians (2π n/L); L=1
    int nx = (ix<=NX/2)? ix : ix-NX;       // FFTW-style wrap
    int ny = (iy<=NY/2)? iy : iy-NY;
    float kx = 2.0f*M_PI*nx;
    float ky = 2.0f*M_PI*ny;
    k2[iy*NX + ix] = kx*kx + ky*ky;
}

// multiply complex field in-place by scale = (-1/k^2) with (0,0)->0
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

// subtract spatial mean of a real field in place (device)
__global__ void subtract_mean(float* __restrict__ f, int N, float mean){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N) f[i] -= mean;
}

// gradient kernel: m_x -= dt * dφ/dx, m_y -= dt * dφ/dy (periodic centered diff)
__global__ void apply_grad_phi(float* __restrict__ mx, float* __restrict__ my,
                               const float* __restrict__ phi,
                               int NX, int NY, float dx, float dy, float dt)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix>=NX || iy>=NY) return;
    auto at = [&](int x,int y)->float{
        x = (x+NX)%NX; y=(y+NY)%NY; return phi[y*NX + x];
    };
    float dphix = (at(ix+1,iy) - at(ix-1,iy)) / (2.0f*dx);
    float dphiy = (at(ix,iy+1) - at(ix,iy-1)) / (2.0f*dy);
    int idx=iy*NX+ix;
    mx[idx] -= dt * dphix;
    my[idx] -= dt * dphiy;
}

// Simple real-to-complex path using full complex grid (C2C forward/back)
// (We could use R2C/C2R to save memory, but C2C keeps the code short.)
struct Poisson2D {
    int NX, NY;
    cufftHandle planF{}, planI{};
    float *d_k2{};        // NX*NY (float)
    cufftComplex *d_tmp{}; // NX*NY (complex scratch in k-space)
    bool ready{false};

    void init(int NX_, int NY_){
        NX=NX_; NY=NY_;
        // complex size = NX*NY
        CHECK_CUDA(cudaMalloc(&d_k2,  NX*NY*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_tmp, NX*NY*sizeof(cufftComplex)));

        dim3 b(16,16), g((NX+15)/16,(NY+15)/16);
        fill_k2<<<g,b>>>(d_k2, NX, NY);
        CHECK_CUDA(cudaDeviceSynchronize());

        // C2C plans (in-place allowed; we’ll copy real→complex first)
        CHECK_CUFFT(cufftPlan2d(&planF, NY, NX, CUFFT_C2C));
        CHECK_CUFFT(cufftPlan2d(&planI, NY, NX, CUFFT_C2C));
        ready=true;
    }
    void destroy(){
        if(!ready) return;
        cufftDestroy(planF); cufftDestroy(planI);
        cudaFree(d_k2); cudaFree(d_tmp);
        ready=false;
    }

    // rhs: real (device) NX*NY
    // phi: real (device) NX*NY
    void solve(const float* rhs_real, float* phi_real, float mean_rho, float dx, float dy){
        // Copy rhs → tmp.real, zero imag
        // Also subtract mean in-place in tmp.real
        // Pack real into complex: two kernels for brevity
        // (1) copy real to phi_real (then subtract mean)
        size_t N = (size_t)NX*NY;
        CHECK_CUDA(cudaMemcpy(phi_real, rhs_real, N*sizeof(float), cudaMemcpyDeviceToDevice));
        // subtract mean so RHS has zero-mean (periodic solvability)
        int tb=256, gb=(N+tb-1)/tb;
        subtract_mean<<<gb,tb>>>(phi_real, (int)N, mean_rho);
        CHECK_CUDA(cudaDeviceSynchronize());

        // (2) real->complex buffer: copy into d_tmp.x, set .y=0
        // reuse a tiny kernel
        auto toComplex = [] __device__ (float r){ return make_cuFloatComplex(r,0.0f); };
        // simple 1D kernel:
        struct K { static __global__ void run(cufftComplex* dst, const float* src, int N){
            int i=blockIdx.x*blockDim.x+threadIdx.x;
            if(i<N) dst[i]=make_cuFloatComplex(src[i],0.0f);
        }}; 
        K::run<<<gb,tb>>>(d_tmp, phi_real, (int)N);
        CHECK_CUDA(cudaDeviceSynchronize());

        // FFT forward
        CHECK_CUFFT(cufftExecC2C(planF, d_tmp, d_tmp, CUFFT_FORWARD));

        // scale by (-1/k^2), set DC to 0
        dim3 b2(16,16), g2((NX+15)/16,(NY+15)/16);
        apply_poisson_scaling<<<g2,b2>>>(d_tmp, d_k2, NX, NY);
        CHECK_CUDA(cudaDeviceSynchronize());

        // inverse FFT
        CHECK_CUFFT(cufftExecC2C(planI, d_tmp, d_tmp, CUFFT_INVERSE));

        // copy back real part and apply normalization (1/N)
        struct K2{ static __global__ void run(float* out, const cufftComplex* in, int N, float invN){
            int i=blockIdx.x*blockDim.x+threadIdx.x;
            if(i<N) out[i] = in[i].x * invN; // imag ~ 0
        }};
        float invN = 1.0f / float(NX*NY);
        K2::run<<<gb,tb>>>(phi_real, d_tmp, (int)N, invN);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
};

// ---- ENERGY SPECTRUM (kinetic) --------------------------------------------
// Compute E_k(x)=0.5*rho*(vx^2+vy^2), FFT it (C2C), make |Ekhat|^2, radial-bin.
struct Spectrum2D {
    int NX, NY;
    cufftHandle planF{};
    cufftComplex* d_c{};   // NX*NY complex buffer
    bool ready{false};

    void init(int NX_, int NY_){
        NX=NX_; NY=NY_;
        CHECK_CUDA(cudaMalloc(&d_c, NX*NY*sizeof(cufftComplex)));
        CHECK_CUFFT(cufftPlan2d(&planF, NY, NX, CUFFT_C2C));
        ready=true;
    }
    void destroy(){ if(!ready) return; cufftDestroy(planF); cudaFree(d_c); ready=false; }

    static __global__ void make_ke_field(cufftComplex* dst,
                                         const float* rho, const float* mx, const float* my,
                                         int N)
    {
        int i=blockIdx.x*blockDim.x+threadIdx.x;
        if(i<N){
            float r=fmaxf(rho[i],1e-6f);
            float vx=mx[i]/r, vy=my[i]/r;
            float Ek=0.5f*r*(vx*vx+vy*vy);
            dst[i]=make_cuFloatComplex(Ek,0.0f);
        }
    }
    static __global__ void power_spectrum_bins(const cufftComplex* F, int NX,int NY,
                                               float* bins, unsigned* counts, int nbins)
    {
        int ix=blockIdx.x*blockDim.x+threadIdx.x;
        int iy=blockIdx.y*blockDim.y+threadIdx.y;
        if(ix>=NX||iy>=NY) return;
        int nx = (ix<=NX/2)? ix : ix-NX;
        int ny = (iy<=NY/2)? iy : iy-NY;
        float k = sqrtf(float(nx*nx + ny*ny));
        int b = min(nbins-1, int(k+0.5f)); // 1 bin ≈ 1 “pixel” in radius
        int idx=iy*NX+ix;
        float re=F[idx].x, im=F[idx].y;
        float p = re*re + im*im;
        atomicAdd(&bins[b], p);
        atomicAdd(&counts[b], 1u);
    }

    // returns host vector with nbins entries
    std::vector<float> compute(const float* d_rho, const float* d_mx, const float* d_my, int nbins){
        size_t N = (size_t)NX*NY;
        int tb=256, gb=(N+tb-1)/tb;
        make_ke_field<<<gb,tb>>>(d_c, d_rho, d_mx, d_my, (int)N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUFFT(cufftExecC2C(planF, d_c, d_c, CUFFT_FORWARD));

        float *d_bins=nullptr; unsigned *d_cnt=nullptr;
        CHECK_CUDA(cudaMalloc(&d_bins, nbins*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_cnt , nbins*sizeof(unsigned)));
        CHECK_CUDA(cudaMemset(d_bins,0, nbins*sizeof(float)));
        CHECK_CUDA(cudaMemset(d_cnt ,0, nbins*sizeof(unsigned)));

        dim3 b(16,16), g((NX+15)/16,(NY+15)/16);
        power_spectrum_bins<<<g,b>>>(d_c,NX,NY,d_bins,d_cnt,nbins);
        CHECK_CUDA(cudaDeviceSynchronize());

        std::vector<float> h_bins(nbins,0.f);
        std::vector<unsigned> h_cnt(nbins,0);
        CHECK_CUDA(cudaMemcpy(h_bins.data(), d_bins, nbins*sizeof(float), cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaMemcpy(h_cnt.data(),  d_cnt,  nbins*sizeof(unsigned), cudaMemcpyDeviceToHost));
        cudaFree(d_bins); cudaFree(d_cnt);

        for(int i=0;i<nbins;i++){
            if(h_cnt[i]>0) h_bins[i] /= float(h_cnt[i]);  // average per mode
        }
        return h_bins;
    }
};
