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

// ---------------- Free (non-member) kernels ----------------
__global__ void fill_k2(float* __restrict__ k2, int NX, int NY){
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if(ix>=NX || iy>=NY) return;
    int nx = (ix<=NX/2)? ix : ix-NX;
    int ny = (iy<=NY/2)? iy : iy-NY;
    float kx = 2.0f*M_PI*nx;
    float ky = 2.0f*M_PI*ny;
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
    if (ix==0 && iy==0){ F[idx].x = 0.0f; F[idx].y = 0.0f; return; }
    float kk = k2[idx];
    float s = (kk>0.f)? (-1.0f/kk) : 0.0f;
    cufftComplex v = F[idx];
    F[idx].x = v.x * s;
    F[idx].y = v.y * s;
}

__global__ void subtract_mean(float* __restrict__ f, int N, float mean){
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i<N) f[i] -= mean;
}

__global__ void copyRealToComplex(cufftComplex* dst, const float* src, int N){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) dst[i]=make_cuFloatComplex(src[i],0.0f);
}

__global__ void complexToRealScaled(float* out, const cufftComplex* in, int N, float invN){
    int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<N) out[i] = in[i].x * invN;
}

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

// For spectrum:
__global__ void make_ke_field(cufftComplex* dst,
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

__global__ void power_spectrum_bins(const cufftComplex* F, int NX,int NY,
                                    float* bins, unsigned* counts, int nbins)
{
    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    if(ix>=NX||iy>=NY) return;
    int nx = (ix<=NX/2)? ix : ix-NX;
    int ny = (iy<=NY/2)? iy : iy-NY;
    float k = sqrtf(float(nx*nx + ny*ny));
    int b = min(nbins-1, int(k+0.5f)); // ~1 “pixel” per radial bin
    int idx=iy*NX+ix;
    float re=F[idx].x, im=F[idx].y;
    float p = re*re + im*im;
    atomicAdd(&bins[b], p);
    atomicAdd(&counts[b], 1u);
}

// ---------------- Poisson & Spectrum helpers ----------------
struct Poisson2D {
    int NX=0, NY=0;
    cufftHandle planF{}, planI{};
    float *d_k2=nullptr;
    cufftComplex *d_tmp=nullptr;
    bool ready=false;

    void init(int NX_, int NY_){
        NX=NX_; NY=NY_;
        CHECK_CUDA(cudaMalloc(&d_k2,  NX*NY*sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_tmp, NX*NY*sizeof(cufftComplex)));

        dim3 b(16,16), g((NX+15)/16,(NY+15)/16);
        fill_k2<<<g,b>>>(d_k2, NX, NY);
        CHECK_CUDA(cudaDeviceSynchronize());

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

    void solve(const float* rhs_real, float* phi_real, float mean_rho, float dx, float dy){
        size_t N = (size_t)NX*NY;
        CHECK_CUDA(cudaMemcpy(phi_real, rhs_real, N*sizeof(float), cudaMemcpyDeviceToDevice));
        int tb=256, gb=(N+tb-1)/tb;
        subtract_mean<<<gb,tb>>>(phi_real, (int)N, mean_rho);
        CHECK_CUDA(cudaDeviceSynchronize());

        copyRealToComplex<<<gb,tb>>>(d_tmp, phi_real, (int)N);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUFFT(cufftExecC2C(planF, d_tmp, d_tmp, CUFFT_FORWARD));

        dim3 b2(16,16), g2((NX+15)/16,(NY+15)/16);
        apply_poisson_scaling<<<g2,b2>>>(d_tmp, d_k2, NX, NY);
        CHECK_CUDA(cudaDeviceSynchronize());

        CHECK_CUFFT(cufftExecC2C(planI, d_tmp, d_tmp, CUFFT_INVERSE));

        float invN = 1.0f / float(NX*NY);
        complexToRealScaled<<<gb,tb>>>(phi_real, d_tmp, (int)N, invN);
        CHECK_CUDA(cudaDeviceSynchronize());
    }
};

struct Spectrum2D {
    int NX=0, NY=0;
    cufftHandle planF{};
    cufftComplex* d_c=nullptr;
    bool ready=false;

    void init(int NX_, int NY_){
        NX=NX_; NY=NY_;
        CHECK_CUDA(cudaMalloc(&d_c, NX*NY*sizeof(cufftComplex)));
        CHECK_CUFFT(cufftPlan2d(&planF, NY, NX, CUFFT_C2C));
        ready=true;
    }
    void destroy(){ if(!ready) return; cufftDestroy(planF); cudaFree(d_c); ready=false; }

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
            if(h_cnt[i]>0) h_bins[i] /= float(h_cnt[i]);
        }
        return h_bins;
    }
};
