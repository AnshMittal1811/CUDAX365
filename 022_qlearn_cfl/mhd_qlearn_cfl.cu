#include <cstdio>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <cstring>
#include "poisson_fft.cuh"

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do { auto err=(x); if (err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)
#endif

constexpr float GAMMA=1.4f, RHO_MIN=1e-6f, P_MIN=1e-6f;
constexpr float PI=3.14159265358979323846f;

__device__ __forceinline__ float safe_rho(float rho){ return fmaxf(rho, RHO_MIN); }
__device__ __forceinline__ float pressure(float rho, float mx, float my, float bx, float by, float E){
    float r=safe_rho(rho), vx=mx/r, vy=my/r;
    float kin=0.5f*r*(vx*vx+vy*vy), mag=0.5f*(bx*bx+by*by);
    return fmaxf((GAMMA-1.f)*(E-kin-mag), P_MIN);
}
__device__ __forceinline__ float fast_magnetosonic(float rho, float p, float bx, float by){
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
    F[0]=mx; F[1]=mx*vx+pt-bx*bx; F[2]=mx*vy-bx*by; F[3]=0.0f;
    F[4]=vy*bx-vx*by; F[5]=(E+pt)*vx-bx*vb;
}
__device__ __forceinline__ void flux_y(const float U[6], float G[6]){
    float rho=U[0], mx=U[1], my=U[2], bx=U[3], by=U[4], E=U[5];
    float r=safe_rho(rho), vx=mx/r, vy=my/r, p=pressure(rho,mx,my,bx,by,E);
    float bt2=bx*bx+by*by, pt=p+0.5f*bt2, vb=vx*bx+vy*by;
    G[0]=my; G[1]=my*vx-by*bx; G[2]=my*vy+pt-by*by; G[3]=vx*by-vy*bx;
    G[4]=0.0f; G[5]=(E+pt)*vy-by*vb;
}
__device__ __forceinline__ void rusanov_flux(const float UL[6], const float UR[6], int dirXY, float F[6]){
    float FL[6], FR[6];
    if (dirXY==0){ flux_x(UL,FL); flux_x(UR,FR); }
    else         { flux_y(UL,FL); flux_y(UR,FR); }
    float rL=safe_rho(UL[0]), rR=safe_rho(UR[0]);
    float vxL=UL[1]/rL, vyL=UL[2]/rL, vxR=UR[1]/rR, vyR=UR[2]/rR;
    float pL=pressure(UL[0],UL[1],UL[2],UL[3],UL[4],UL[5]);
    float pR=pressure(UR[0],UR[1],UR[2],UR[3],UR[4],UR[5]);
    float cfL=fast_magnetosonic(UL[0],pL,UL[3],UL[4]);
    float cfR=fast_magnetosonic(UR[0],pR,UR[3],UR[4]);
    float vnL=(dirXY==0)?vxL:vyL, vnR=(dirXY==0)?vxR:vyR;
    float smax=fmaxf(fabsf(vnL)+cfL, fabsf(vnR)+cfR);
    #pragma unroll
    for(int m=0;m<6;m++) F[m]=0.5f*(FL[m]+FR[m]) - 0.5f*smax*(UR[m]-UL[m]);
}

__global__ void step_mhd_rusanov(const float* Uin, float* Uout,
                                 int NX,int NY,float dx,float dy,float dt)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=NX || iy>=NY) return;
    int idx = iy*NX + ix;
    int im1=wrap(ix-1,NX), ip1=wrap(ix+1,NX);
    int jm1=wrap(iy-1,NY), jp1=wrap(iy+1,NY);
    float Uc[6], Ul[6], Ur[6], Ud[6], Uu[6];
    #pragma unroll
    for(int c=0;c<6;c++){
        Uc[c]=Uin[c*NX*NY + idx];
        Ul[c]=Uin[c*NX*NY + iy*NX + im1];
        Ur[c]=Uin[c*NX*NY + iy*NX + ip1];
        Ud[c]=Uin[c*NX*NY + jm1*NX + ix];
        Uu[c]=Uin[c*NX*NY + jp1*NX + ix];
    }
    float FxL[6], FxR[6], GyD[6], GyU[6];
    rusanov_flux(Ul,Uc,0,FxL); rusanov_flux(Uc,Ur,0,FxR);
    rusanov_flux(Ud,Uc,1,GyD); rusanov_flux(Uc,Uu,1,GyU);
    float dtdx=dt/dx, dtdy=dt/dy;
    #pragma unroll
    for(int c=0;c<6;c++){
        float Un = Uc[c] - dtdx*(FxR[c]-FxL[c]) - dtdy*(GyU[c]-GyD[c]);
        Uout[c*NX*NY + idx] = Un;
    }
}

__device__ unsigned int d_max_speed_bits;
__global__ void reset_maxspeed(){ d_max_speed_bits = 0u; }
__device__ __forceinline__ void atomicMaxFloatPos(float v){ atomicMax(&d_max_speed_bits, __float_as_uint(v)); }
__global__ void kernel_maxspeed(const float* U, int NX, int NY){
    int ix=blockIdx.x*blockDim.x + threadIdx.x;
    int iy=blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=NX || iy>=NY) return;
    int idx=iy*NX+ix;
    float rho=U[0*NX*NY + idx], mx=U[1*NX*NY + idx], my=U[2*NX*NY + idx];
    float bx =U[3*NX*NY + idx], by=U[4*NX*NY + idx], E =U[5*NX*NY + idx];
    float r  = safe_rho(rho);
    float vx = mx/r, vy=my/r;
    float p  = pressure(rho,mx,my,bx,by,E);
    float cf = fast_magnetosonic(rho,p,bx,by);
    float smax = fmaxf(fabsf(vx)+cf, fabsf(vy)+cf);
    atomicMaxFloatPos(fmaxf(smax,0.f));
}
static inline float uint_as_float_host(uint32_t u){ float f; std::memcpy(&f,&u,sizeof(float)); return f; }
float compute_dt_gpu(const float* dU, int NX, int NY, float dx, float dy, float cfl){
    dim3 block(16,16), grid((NX+block.x-1)/block.x,(NY+block.y-1)/block.y);
    reset_maxspeed<<<1,1>>>();
    kernel_maxspeed<<<grid,block>>>(dU,NX,NY);
    CHECK_CUDA(cudaDeviceSynchronize());
    uint32_t hbits=0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&hbits,d_max_speed_bits,sizeof(uint32_t)));
    float smax = uint_as_float_host(hbits);
    if (!std::isfinite(smax) || smax<=0.f) smax = 1.0f;
    return cfl * fminf(dx,dy) / smax;
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

int main(int argc,char** argv){
    int NX=128, NY=128, STEPS=50;
    float cfl = 0.25f;
    for (int i=1;i<argc;i++){
        if (std::strcmp(argv[i], "--cfl") == 0 && i+1 < argc){ cfl = std::atof(argv[++i]); }
        else if (NX==128) NX = std::atoi(argv[i]);
        else if (NY==128) NY = std::atoi(argv[i]);
        else STEPS = std::atoi(argv[i]);
    }

    float Lx=1.0f, Ly=1.0f, dx=Lx/NX, dy=Ly/NY;
    size_t cells=(size_t)NX*NY, bytes=6*cells*sizeof(float);
    std::vector<float> hU(6*cells);
    init_orszag_tang(hU,NX,NY);
    float *dU=nullptr,*dV=nullptr,*d_phi=nullptr;
    CHECK_CUDA(cudaMalloc(&dU,bytes));
    CHECK_CUDA(cudaMalloc(&dV,bytes));
    CHECK_CUDA(cudaMalloc(&d_phi,cells*sizeof(float)));
    CHECK_CUDA(cudaMemcpy(dU,hU.data(),bytes,cudaMemcpyHostToDevice));
    Poisson2D poisson; poisson.init(NX,NY);
    dim3 block(16,16), grid((NX+block.x-1)/block.x,(NY+block.y-1)/block.y);

    double E0=0.0;
    for (size_t i=0;i<cells;i++) E0 += hU[5*cells + i];

    for (int s=0; s<STEPS; ++s){
        float dt = compute_dt_gpu(dU,NX,NY,dx,dy,cfl);
        step_mhd_rusanov<<<grid,block>>>(dU,dV,NX,NY,dx,dy,dt);
        CHECK_CUDA(cudaDeviceSynchronize());
        std::swap(dU,dV);
        // mean rho on host
        std::vector<float> h_rho(cells);
        CHECK_CUDA(cudaMemcpy(h_rho.data(), dU, cells*sizeof(float), cudaMemcpyDeviceToHost));
        double sum=0.0; for(size_t i=0;i<cells;i++) sum += h_rho[i];
        float mean_rho = float(sum / double(cells));
        poisson.solve(dU + 0*cells, d_phi, mean_rho, dx, dy);
        apply_grad_phi<<<grid,block>>>(dU + 1*cells, dU + 2*cells, d_phi, NX, NY, dx, dy, dt);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    std::vector<float> hUout(6*cells);
    CHECK_CUDA(cudaMemcpy(hUout.data(), dU, bytes, cudaMemcpyDeviceToHost));
    double E1=0.0; int nan=0;
    for (size_t i=0;i<cells;i++){
        float v = hUout[5*cells + i];
        E1 += v;
        if (!std::isfinite(hUout[i])) nan = 1;
    }
    double drift = (E1 - E0) / (E0 + 1e-12);
    printf("Energy drift: %.6e\n", drift);
    printf("NaN: %d\n", nan);

    poisson.destroy();
    cudaFree(dU); cudaFree(dV); cudaFree(d_phi);
    return 0;
}
