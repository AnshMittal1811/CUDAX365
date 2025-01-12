#include <cstdio>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>
#include "poisson_fft.cuh"

#define CHECK_CUDA(x) do { auto err=(x); if (err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)

constexpr float GAMMA=1.4f, RHO_MIN=1e-6f, P_MIN=1e-6f, CFL=0.30f;
#ifndef TILE_X
#define TILE_X 16
#endif
#ifndef TILE_Y
#define TILE_Y 16
#endif
#ifndef PAD_X
#define PAD_X 1
#endif
#ifndef APPLY_POISSON_GRAD
#define APPLY_POISSON_GRAD 1   // set 0 to skip momentum correction
#endif

__device__ __forceinline__ float safe_rho(float rho){ return fmaxf(rho, RHO_MIN); }
__device__ __forceinline__ float pressure(float rho, float mx, float my, float bx, float by, float E){
    float r=safe_rho(rho), vx=mx/r, vy=my/r;
    float kin=0.5f*r*(vx*vx+vy*vy), mag=0.5f*(bx*bx+by*by);
    return fmaxf((GAMMA-1.f)*(E-kin-mag), P_MIN);
}
__device__ __forceinline__ float cf_fast(float rho,float p,float bx,float by){
    float r=safe_rho(rho);
    float cs2=fmaxf(GAMMA*p/r,1e-12f);
    float vA2=(bx*bx+by*by)/r;
    return sqrtf(cs2+vA2);
}
__device__ __forceinline__ int wrap(int i,int n){ i%=n; if(i<0) i+=n; return i; }

__device__ __forceinline__ void flux_x(const float U[6], float F[6]){
    float rho=U[0], mx=U[1], my=U[2], bx=U[3], by=U[4], E=U[5];
    float r=safe_rho(rho), vx=mx/r, vy=my/r, p=pressure(rho,mx,my,bx,by,E);
    float bt2=bx*bx+by*by, pt=p+0.5f*bt2, vb=vx*bx+vy*by;
    F[0]=mx; F[1]=mx*vx+pt-bx*bx; F[2]=mx*vy-bx*by; F[3]=0.0f; F[4]=vy*bx-vx*by; F[5]=(E+pt)*vx-bx*vb;
}
__device__ __forceinline__ void flux_y(const float U[6], float G[6]){
    float rho=U[0], mx=U[1], my=U[2], bx=U[3], by=U[4], E=U[5];
    float r=safe_rho(rho), vx=mx/r, vy=my/r, p=pressure(rho,mx,my,bx,by,E);
    float bt2=bx*bx+by*by, pt=p+0.5f*bt2, vb=vx*bx+vy*by;
    G[0]=my; G[1]=my*vx-by*bx; G[2]=my*vy+pt-by*by; G[3]=vx*by-vy*bx; G[4]=0.0f; G[5]=(E+pt)*vy-by*vb;
}
__device__ __forceinline__ void rusanov(const float UL[6], const float UR[6], int dirXY, float F[6]){
    float FL[6], FR[6];
    if (dirXY==0){ flux_x(UL,FL); flux_x(UR,FR); }
    else         { flux_y(UL,FL); flux_y(UR,FR); }
    float rL=safe_rho(UL[0]), rR=safe_rho(UR[0]);
    float vxL=UL[1]/rL, vyL=UL[2]/rL, vxR=UR[1]/rR, vyR=UR[2]/rR;
    float pL=pressure(UL[0],UL[1],UL[2],UL[3],UL[4],UL[5]);
    float pR=pressure(UR[0],UR[1],UR[2],UR[3],UR[4],UR[5]);
    float cfL=cf_fast(UL[0],pL,UL[3],UL[4]), cfR=cf_fast(UR[0],pR,UR[3],UR[4]);
    float vnL=(dirXY==0)?vxL:vyL, vnR=(dirXY==0)?vxR:vyR;
    float smax=fmaxf(fabsf(vnL)+cfL, fabsf(vnR)+cfR);
    #pragma unroll
    for(int m=0;m<6;m++) F[m]=0.5f*(FL[m]+FR[m]) - 0.5f*smax*(UR[m]-UL[m]);
}

// --- tiled kernel (unchanged numerics) ---
template<int TX,int TY,int PAD>
__global__ void step_mhd_tiled(const float* __restrict__ Uin, float* __restrict__ Uout,
                               int NX,int NY,float dx,float dy,float dt){
    constexpr int SX = TX + 2 + PAD;
    constexpr int SY = TY + 2;
    __shared__ float sh_rho[SY][SX], sh_mx[SY][SX], sh_my[SY][SX],
                     sh_bx[SY][SX],  sh_by[SY][SX], sh_E [SY][SX];

    int ix = blockIdx.x*TX + threadIdx.x;
    int iy = blockIdx.y*TY + threadIdx.y;
    int tx = threadIdx.x + 1, ty = threadIdx.y + 1;

    auto gAt=[&](int c,int x,int y)->float{
        int idx=y*NX+x; const float* U=Uin;
        return U[c*(size_t)NX*NY + idx];
    };
    if (ix<NX && iy<NY){
        sh_rho[ty][tx]=gAt(0,ix,iy); sh_mx[ty][tx]=gAt(1,ix,iy);
        sh_my[ty][tx]=gAt(2,ix,iy);  sh_bx[ty][tx]=gAt(3,ix,iy);
        sh_by[ty][tx]=gAt(4,ix,iy);  sh_E [ty][tx]=gAt(5,ix,iy);
    }
    if (threadIdx.x==0){
        int gx=wrap(ix-1,NX), gy=iy;
        if(gy<NY){ sh_rho[ty][0]=gAt(0,gx,gy); sh_mx[ty][0]=gAt(1,gx,gy);
                   sh_my[ty][0]=gAt(2,gx,gy);  sh_bx[ty][0]=gAt(3,gx,gy);
                   sh_by[ty][0]=gAt(4,gx,gy);  sh_E [ty][0]=gAt(5,gx,gy); }
    }
    if (threadIdx.x==TX-1){
        int gx=wrap(ix+1,NX), gy=iy;
        if(gy<NY){ sh_rho[ty][TX+1]=gAt(0,gx,gy); sh_mx[ty][TX+1]=gAt(1,gx,gy);
                   sh_my[ty][TX+1]=gAt(2,gx,gy);  sh_bx[ty][TX+1]=gAt(3,gx,gy);
                   sh_by[ty][TX+1]=gAt(4,gx,gy);  sh_E [ty][TX+1]=gAt(5,gx,gy); }
    }
    if (threadIdx.y==0){
        int gx=ix, gy=wrap(iy-1,NY);
        if(gx<NX){ sh_rho[0][tx]=gAt(0,gx,gy); sh_mx[0][tx]=gAt(1,gx,gy);
                   sh_my[0][tx]=gAt(2,gx,gy);  sh_bx[0][tx]=gAt(3,gx,gy);
                   sh_by[0][tx]=gAt(4,gx,gy);  sh_E [0][tx]=gAt(5,gx,gy); }
    }
    if (threadIdx.y==TY-1){
        int gx=ix, gy=wrap(iy+1,NY);
        if(gx<NX){ sh_rho[TY+1][tx]=gAt(0,gx,gy); sh_mx[TY+1][tx]=gAt(1,gx,gy);
                   sh_my[TY+1][tx]=gAt(2,gx,gy);  sh_bx[TY+1][tx]=gAt(3,gx,gy);
                   sh_by[TY+1][tx]=gAt(4,gx,gy);  sh_E [TY+1][tx]=gAt(5,gx,gy); }
    }
    __syncthreads();
    if (ix>=NX || iy>=NY) return;

    float Uc[6], ULx[6], URx[6], UDy[6], UUy[6];
    #pragma unroll
    for(int c=0;c<6;c++){
        const float (*pp)[SX]=(c==0?sh_rho:c==1?sh_mx:c==2?sh_my:c==3?sh_bx:c==4?sh_by:sh_E);
        Uc[c]=pp[ty][tx]; ULx[c]=pp[ty][tx-1]; URx[c]=pp[ty][tx+1];
        UDy[c]=pp[ty-1][tx]; UUy[c]=pp[ty+1][tx];
    }

    float FxL[6], FxR[6], GyD[6], GyU[6];
    rusanov(ULx, Uc , 0, FxL); rusanov(Uc , URx, 0, FxR);
    rusanov(UDy, Uc , 1, GyD); rusanov(Uc , UUy, 1, GyU);

    float Un[6];
    #pragma unroll
    for(int c=0;c<6;c++){
        float dUx=-(FxR[c]-FxL[c]);
        float dUy=-(GyU[c]-GyD[c]);
        Un[c]=Uc[c]; // accumulate after scale
        // defer /dx,/dy until write:
        Un[c] = Uc[c] + (dUx) + (dUy);
    }
    // write scaled by dt/(dx,dy):
    size_t idx=(size_t)iy*NX+ix;
    #pragma unroll
    for(int c=0;c<6;c++)
        Un[c]=Uc[c] + (-(FxR[c]-FxL[c])/dx - (GyU[c]-GyD[c])/dy) * dt;

    Un[0]=fmaxf(Un[0], RHO_MIN);
    {
        float p=pressure(Un[0],Un[1],Un[2],Un[3],Un[4],Un[5]);
        float r=safe_rho(Un[0]), vx=Un[1]/r, vy=Un[2]/r;
        float kin=0.5f*r*(vx*vx+vy*vy), mag=0.5f*(Un[3]*Un[3]+Un[4]*Un[4]);
        float Emin=kin+mag+P_MIN/(GAMMA-1.f);
        if (Un[5]<Emin) Un[5]=Emin;
    }

    #pragma unroll
    for(int c=0;c<6;c++) Uout[c*(size_t)NX*NY + idx]=Un[c];
}

// --- tiny CFL (same as Day 11) ---
__device__ unsigned int d_max_speed_bits;
__global__ void reset_maxspeed(){ d_max_speed_bits=0u; }
__device__ __forceinline__ void atomicMaxFloatPos(float v){ atomicMax(&d_max_speed_bits, __float_as_uint(v)); }
__global__ void kernel_maxspeed(const float* U,int NX,int NY){
    int ix=blockIdx.x*blockDim.x+threadIdx.x, iy=blockIdx.y*blockDim.y+threadIdx.y;
    if (ix>=NX||iy>=NY) return;
    size_t i=(size_t)iy*NX+ix;
    float rho=U[0*(size_t)NX*NY+i], mx=U[1*(size_t)NX*NY+i], my=U[2*(size_t)NX*NY+i];
    float bx =U[3*(size_t)NX*NY+i], by=U[4*(size_t)NX*NY+i], E =U[5*(size_t)NX*NY+i];
    auto safe=[&](float r){return fmaxf(r,1e-6f);};
    float r=safe(rho), vx=mx/r, vy=my/r;
    float kin=0.5f*r*(vx*vx+vy*vy), mag=0.5f*(bx*bx+by*by);
    float p=fmaxf((GAMMA-1.f)*(E-kin-mag), P_MIN);
    float cf = sqrtf(fmaxf(GAMMA*p/r,1e-12f) + (bx*bx+by*by)/r);
    atomicMaxFloatPos(fmaxf(fabsf(vx)+cf, fabsf(vy)+cf));
}
static inline float uint_as_float_host(unsigned u){ float f; memcpy(&f,&u,4); return f; }
float compute_dt_gpu(const float* dU,int NX,int NY,float dx,float dy){
    dim3 b(16,16), g((NX+15)/16,(NY+15)/16);
    reset_maxspeed<<<1,1>>>(); kernel_maxspeed<<<g,b>>>(dU,NX,NY);
    CHECK_CUDA(cudaDeviceSynchronize());
    unsigned h=0; CHECK_CUDA(cudaMemcpyFromSymbol(&h,d_max_speed_bits,sizeof(h)));
    float smax=uint_as_float_host(h); if (!(isfinite(smax)&&smax>0.f)) smax=1.0f;
    return CFL * fminf(dx,dy) / smax;
}

// --- init Orszag–Tang ---
void init_orszag_tang(std::vector<float>& hU,int NX,int NY,float Lx,float Ly){
    auto at=[&](int c,int x,int y)->float&{ return hU[c*(size_t)NX*NY + (size_t)y*NX + x]; };
    for(int y=0;y<NY;y++) for(int x=0;x<NX;x++){
        float X=(x+0.5f)*Lx/NX, Y=(y+0.5f)*Ly/NY;
        float rho=1.f, vx=-sinf(2.f*M_PI*Y), vy=sinf(2.f*M_PI*X);
        float bx=-sinf(2.f*M_PI*Y), by=sinf(4.f*M_PI*X), p=1.f;
        float mx=rho*vx, my=rho*vy, kin=0.5f*rho*(vx*vx+vy*vy), mag=0.5f*(bx*bx+by*by);
        float E=p/(GAMMA-1.f)+kin+mag;
        at(0,x,y)=rho; at(1,x,y)=mx; at(2,x,y)=my;
        at(3,x,y)=bx;  at(4,x,y)=by; at(5,x,y)=E;
    }
}

int main(int argc,char**argv){
    int NX=(argc>1)?atoi(argv[1]):256;
    int NY=(argc>2)?atoi(argv[2]):256;
    int STEPS=(argc>3)?atoi(argv[3]):200;

    float Lx=1.f,Ly=1.f, dx=Lx/NX, dy=Ly/NY;
    size_t cells=(size_t)NX*NY, bytes=6*cells*sizeof(float);

    std::vector<float> hU(6*cells);
    init_orszag_tang(hU,NX,NY,Lx,Ly);

    float *dU=nullptr,*dV=nullptr;
    CHECK_CUDA(cudaMalloc(&dU,bytes));
    CHECK_CUDA(cudaMalloc(&dV,bytes));
    CHECK_CUDA(cudaMemcpy(dU,hU.data(),bytes,cudaMemcpyHostToDevice));

    // <<< POISSON >>> init helpers
    Poisson2D poisson; poisson.init(NX,NY);
    Spectrum2D spec;   spec.init(NX,NY);
    float *d_phi=nullptr; CHECK_CUDA(cudaMalloc(&d_phi, cells*sizeof(float)));

    dim3 block(TILE_X,TILE_Y);
    dim3 grid((NX+TILE_X-1)/TILE_X,(NY+TILE_Y-1)/TILE_Y);

    for (int s=0; s<STEPS; ++s){
        float dt=compute_dt_gpu(dU,NX,NY,dx,dy);
        step_mhd_tiled<TILE_X,TILE_Y,PAD_X><<<grid,block>>>(dU,dV,NX,NY,dx,dy,dt);
        CHECK_CUDA(cudaPeekAtLastError());
        std::swap(dU,dV);

        // <<< POISSON >>> build φ from current ρ and subtract mean(ρ)
        // compute mean(ρ) on host (cheap)
        std::vector<float> tmpRho(cells);
        CHECK_CUDA(cudaMemcpy(tmpRho.data(), dU, cells*sizeof(float), cudaMemcpyDeviceToHost));
        double sum=0.0; for(size_t i=0;i<cells;i++) sum += tmpRho[i];
        float mean_rho = float(sum / double(cells));
        poisson.solve(/*rhs=*/dU, /*phi=*/d_phi, mean_rho, dx, dy);

    #if APPLY_POISSON_GRAD
        // apply m -= dt ∇φ
        dim3 b2(16,16), g2((NX+15)/16,(NY+15)/16);
        apply_grad_phi<<<g2,b2>>>(/*mx*/dU + 1*cells, /*my*/dU + 2*cells,
                                  d_phi, NX, NY, dx, dy, dt);
        CHECK_CUDA(cudaDeviceSynchronize());
    #endif

        if ((s%50)==0){
            // <<< SPECTRUM >>> print 8 first bins
            auto H = spec.compute(/*rho*/dU + 0*cells, /*mx*/dU + 1*cells, /*my*/dU + 2*cells, /*nbins=*/64);
            printf("step %d/%d (dt=%.3e) spectrum[0..7]:", s,STEPS,dt);
            for(int i=0;i<8;i++) printf(" %.3e", H[i]);
            printf("\n");
        }
    }
    poisson.destroy(); spec.destroy();
    cudaFree(d_phi);

    CHECK_CUDA(cudaMemcpy(hU.data(),dU,bytes,cudaMemcpyDeviceToHost));
    int cx=NX/2, cy=NY/2;
    auto at=[&](int c,int x,int y){ return hU[c*(size_t)NX*NY + (size_t)y*NX + x]; };
    printf("Center: rho=%.5f mx=%.5f my=%.5f Bx=%.5f By=%.5f E=%.5f\n",
           at(0,cx,cy),at(1,cx,cy),at(2,cx,cy),at(3,cx,cy),at(4,cx,cy),at(5,cx,cy));
    cudaFree(dU); cudaFree(dV);
    return 0;
}
