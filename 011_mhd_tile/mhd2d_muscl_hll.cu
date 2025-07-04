#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <algorithm>

#define CHECK_CUDA(x) do { auto err=(x); if (err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)

constexpr float GAMMA   = 1.4f;
constexpr float RHO_MIN = 1e-6f;
constexpr float P_MIN   = 1e-6f;
constexpr float CFL     = 0.30f;

// toggles
#ifndef USE_MUSCL
#define USE_MUSCL 1
#endif
#ifndef USE_POWELL
#define USE_POWELL 1
#endif
#ifndef USE_HLL
#define USE_HLL 1      // <<<<< set 1 to use HLL, 0 to use Rusanov
#endif

// ---------- math helpers ----------
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

__device__ __forceinline__ float minmod(float a, float b){
    if (a*b <= 0.f) return 0.f;
    return (fabsf(a) < fabsf(b)) ? a : b;
}
__device__ __forceinline__ float minmod3(float a, float b, float c){
    return minmod(a, minmod(b,c));
}

// ---------- fluxes ----------
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
    F[3] = 0.0f;                // (CT would put -∂_x ψ here)
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

// ---------- Riemann solvers ----------
// Rusanov (already in your code)
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

// HLL: use two-wave bounds sL, sR from (v_n ± c_f) on both sides.
__device__ __forceinline__ void hll_flux(
    const float UL[6], const float UR[6], int dirXY, float outF[6])
{
    float FL[6], FR[6];
    if (dirXY==0){ flux_x(UL, FL); flux_x(UR, FR); }
    else         { flux_y(UL, FL); flux_y(UR, FR); }

    // normal velocities and c_f on each side
    float rL = safe_rho(UL[0]), rR = safe_rho(UR[0]);
    float vxL=UL[1]/rL, vyL=UL[2]/rL, vxR=UR[1]/rR, vyR=UR[2]/rR;
    float pL = pressure(UL[0],UL[1],UL[2],UL[3],UL[4],UL[5]);
    float pR = pressure(UR[0],UR[1],UR[2],UR[3],UR[4],UR[5]);
    float cfL = fast_magnetosonic(UL[0],pL,UL[3],UL[4]);
    float cfR = fast_magnetosonic(UR[0],pR,UR[3],UR[4]);

    float vnL = (dirXY==0) ? vxL : vyL;
    float vnR = (dirXY==0) ? vxR : vyR;

    float sL = fminf(vnL - cfL, vnR - cfR);
    float sR = fmaxf(vnL + cfL, vnR + cfR);

    if (sL >= 0.0f){
        #pragma unroll
        for (int m=0;m<6;m++) outF[m] = FL[m];
        return;
    }
    if (sR <= 0.0f){
        #pragma unroll
        for (int m=0;m<6;m++) outF[m] = FR[m];
        return;
    }
    float inv = 1.0f / (sR - sL);
    #pragma unroll
    for (int m=0;m<6;m++){
        outF[m] = (sR*FL[m] - sL*FR[m] + sL*sR*(UR[m] - UL[m])) * inv;
    }
}

// ---------- indexing & periodic ----------
__device__ __forceinline__ int wrap(int i, int n){ i%=n; if(i<0) i+=n; return i; }
struct View {
    const float* U; int NX, NY;
    __device__ __forceinline__ float at(int c,int x,int y) const {
        return U[c*NX*NY + y*NX + x];
    }
};

// ---------- MUSCL recon (primitives) ----------
__device__ __forceinline__ void cons_to_prim(const float U[6], float P[6]){
    float rho=U[0], mx=U[1], my=U[2], bx=U[3], by=U[4], E=U[5];
    float r  = safe_rho(rho);
    float vx = mx/r, vy = my/r;
    float p  = pressure(rho,mx,my,bx,by,E);
    P[0]=r; P[1]=vx; P[2]=vy; P[3]=bx; P[4]=by; P[5]=p;
}
__device__ __forceinline__ void prim_to_cons(const float P[6], float U[6]){
    float r=P[0], vx=P[1], vy=P[2], bx=P[3], by=P[4], p=P[5];
    float mx = r*vx, my=r*vy;
    float kin = 0.5f * r * (vx*vx + vy*vy);
    float mag = 0.5f * (bx*bx + by*by);
    float E   = p/(GAMMA-1.f) + kin + mag;
    U[0]=r; U[1]=mx; U[2]=my; U[3]=bx; U[4]=by; U[5]=E;
}

__device__ __forceinline__ void recon_face_1D_prim(
    float PL[6], float PC[6], float PR[6],
    float theta,
    float P_L_on_face[6], float P_R_on_face[6])
{
    #pragma unroll
    for (int k=0;k<6;k++){
        float dl = PC[k]-PL[k];
        float dr = PR[k]-PC[k];
        float dc = 0.5f*(PR[k]-PL[k]);
        float s  = minmod3(dl, dr, theta*dc);
        P_L_on_face[k] = PC[k] + 0.5f*s;
        P_R_on_face[k] = PC[k] - 0.5f*s;
    }
}

// ---------- main step kernel ----------
__global__ void step_mhd_muscl_powell(
    const float* __restrict__ Uin,
    float* __restrict__ Uout,
    int NX, int NY, float dx, float dy, float dt, float theta)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=NX || iy>=NY) return;
    View V{Uin,NX,NY};

    int im1 = wrap(ix-1,NX), ip1 = wrap(ix+1,NX);
    int jm1 = wrap(iy-1,NY), jp1 = wrap(iy+1,NY);

    float ULc[6], UCc[6], URc[6], DLc[6], DCc[6], URy[6];
    #pragma unroll
    for(int c=0;c<6;c++){
        ULc[c]=V.at(c,im1,iy); UCc[c]=V.at(c,ix,iy); URc[c]=V.at(c,ip1,iy);
        DLc[c]=V.at(c,ix,jm1); DCc[c]=V.at(c,ix,iy); URy[c]=V.at(c,ix,jp1);
    }

    float PL[6], PC[6], PR[6], PDL[6], PDC[6], PUR[6];
    cons_to_prim(ULc,PL); cons_to_prim(UCc,PC); cons_to_prim(URc,PR);
    cons_to_prim(DLc,PDL); cons_to_prim(DCc,PDC); cons_to_prim(URy,PUR);

    float P_xR[6], P_xL[6], P_yU[6], P_yD[6];
    recon_face_1D_prim(PL,PC,PR, theta, P_xR, P_xL);
    recon_face_1D_prim(PDL,PDC,PUR,theta, P_yU, P_yD);

    float ULx[6], URx[6], ULy[6], URy_c[6];
    prim_to_cons(P_xR, ULx);
    prim_to_cons(P_xL, URx);
    prim_to_cons(P_yU, ULy);
    prim_to_cons(P_yD, URy_c);

    float FxL[6], FxR[6], GyD[6], GyU[6];

    // Left face (i-1/2): quick mirrored stencil for starter
    {
        float Pm2[6];
        #pragma unroll
        for(int k=0;k<6;k++) Pm2[k] = PL[k] - (PR[k]-PC[k]);
        float PLfaceR[6], PLfaceL[6], ULxL[6], URxL[6];
        recon_face_1D_prim(Pm2, PL, PC, theta, PLfaceR, PLfaceL);
        prim_to_cons(PLfaceR, ULxL); prim_to_cons(PLfaceL, URxL);
    #if USE_HLL
        hll_flux(ULxL, URxL, 0, FxL);
    #else
        rusanov_flux(ULxL, URxL, 0, FxL);
    #endif
    }
    // Right face (i+1/2)
#if USE_HLL
    hll_flux(ULx, URx, 0, FxR);
#else
    rusanov_flux(ULx, URx, 0, FxR);
#endif

    // Bottom face (j-1/2)
    {
        float Pjm2[6];
        #pragma unroll
        for(int k=0;k<6;k++) Pjm2[k] = PDL[k] - (PUR[k]-PDC[k]);
        float PDfaceU[6], PDfaceD[6], ULyB[6], URyB[6];
        recon_face_1D_prim(Pjm2, PDL, PDC, theta, PDfaceU, PDfaceD);
        prim_to_cons(PDfaceU, ULyB); prim_to_cons(PDfaceD, URyB);
    #if USE_HLL
        hll_flux(ULyB, URyB, 1, GyD);
    #else
        rusanov_flux(ULyB, URyB, 1, GyD);
    #endif
    }
    // Top face (j+1/2)
#if USE_HLL
    hll_flux(ULy, URy_c, 1, GyU);
#else
    rusanov_flux(ULy, URy_c, 1, GyU);
#endif

    float Uc[6];
    #pragma unroll
    for(int c=0;c<6;c++) Uc[c]=UCc[c];

    float Un[6];
    #pragma unroll
    for(int c=0;c<6;c++){
        float dUx = -(FxR[c] - FxL[c]);
        float dUy = -(GyU[c] - GyD[c]);
        Un[c] = Uc[c] + ( (dUx) + (dUy) ) * ( (c==0||c==1||c==2||c==3||c==4||c==5) ? 1.0f : 1.0f ); // keep structure; scale below
    }
    // scale by dx,dy
    #pragma unroll
    for(int c=0;c<6;c++){
        Un[c] = Uc[c] + ( -(FxR[c]-FxL[c])/dx - (GyU[c]-GyD[c])/dy ) * dt;
    }

#if USE_POWELL
    // Powell source: -(∇·B)*[0, Bx, By, vx, vy, v·B]
    float bx_im1 = V.at(3,wrap(ix-1,NX),iy), bx_ip1 = V.at(3,wrap(ix+1,NX),iy);
    float by_jm1 = V.at(4,ix,wrap(iy-1,NY)), by_jp1 = V.at(4,ix,wrap(iy+1,NY));
    float divB = (bx_ip1 - bx_im1)/(2.f*dx) + (by_jp1 - by_jm1)/(2.f*dy);

    float r = safe_rho(Un[0]);
    float vx = Un[1]/r, vy = Un[2]/r;
    float vb = vx*Un[3] + vy*Un[4];

    Un[1] += -dt * divB * Un[3];
    Un[2] += -dt * divB * Un[4];
    Un[3] += -dt * divB * vx;
    Un[4] += -dt * divB * vy;
    Un[5] += -dt * divB * vb;
#endif

    // Positivity floors
    Un[0] = fmaxf(Un[0], RHO_MIN);
    {
        float p = pressure(Un[0],Un[1],Un[2],Un[3],Un[4],Un[5]);
        float r = safe_rho(Un[0]);
        float vx=Un[1]/r, vy=Un[2]/r;
        float kin=0.5f*r*(vx*vx+vy*vy);
        float mag=0.5f*(Un[3]*Un[3]+Un[4]*Un[4]);
        float Emin=kin + mag + P_MIN/(GAMMA-1.f);
        if (Un[5] < Emin) Un[5]=Emin;
    }

    int idx = iy*NX + ix;
    #pragma unroll
    for(int c=0;c<6;c++) Uout[c*NX*NY + idx] = Un[c];
}

// ---------- max wavespeed (CFL) ----------
__device__ unsigned int d_max_speed_bits;

__global__ void reset_maxspeed(){ d_max_speed_bits = 0u; }

__device__ __forceinline__ void atomicMaxFloatPos(float v){
    atomicMax(&d_max_speed_bits, __float_as_uint(v));
}

__global__ void kernel_maxspeed(const float* __restrict__ U, int NX, int NY){
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

static inline float uint_as_float_host(uint32_t u){
    float f; std::memcpy(&f,&u,sizeof(float)); return f;
}

float compute_dt_gpu(const float* dU, int NX, int NY, float dx, float dy){
    dim3 block(16,16), grid((NX+block.x-1)/block.x,(NY+block.y-1)/block.y);
    reset_maxspeed<<<1,1>>>();
    kernel_maxspeed<<<grid,block>>>(dU,NX,NY);
    CHECK_CUDA(cudaDeviceSynchronize());
    uint32_t hbits=0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&hbits,d_max_speed_bits,sizeof(uint32_t)));
    float smax = uint_as_float_host(hbits);
    if (!std::isfinite(smax) || smax<=0.f) smax = 1.0f;
    return CFL * fminf(dx,dy) / smax;
}

// ---------- invariants (mass, energy) reduction ----------
__global__ void reduce_mass_energy(
    const float* __restrict__ U, int N, double* out_mass, double* out_energy)
{
    extern __shared__ double sm[];
    double* s_mass = sm;
    double* s_energy = sm + blockDim.x;

    double mass = 0.0, energy = 0.0;
    for (int i=blockIdx.x*blockDim.x + threadIdx.x; i<N; i+=gridDim.x*blockDim.x){
        mass   += (double)U[i]; // rho
        energy += (double)U[5*N + i];
    }
    s_mass[threadIdx.x]   = mass;
    s_energy[threadIdx.x] = energy;
    __syncthreads();

    for (int s=blockDim.x/2; s>0; s>>=1){
        if (threadIdx.x < s){
            s_mass[threadIdx.x]   += s_mass[threadIdx.x + s];
            s_energy[threadIdx.x] += s_energy[threadIdx.x + s];
        }
        __syncthreads();
    }
    if (threadIdx.x==0){
        atomicAdd(out_mass,   s_mass[0]);
        atomicAdd(out_energy, s_energy[0]);
    }
}

void compute_invariants_gpu(const float* dU, int NX, int NY, double& M, double& E){
    int N = NX*NY;
    double *dM=nullptr, *dE=nullptr;
    CHECK_CUDA(cudaMalloc(&dM,sizeof(double)));
    CHECK_CUDA(cudaMalloc(&dE,sizeof(double)));
    CHECK_CUDA(cudaMemset(dM,0, sizeof(double)));
    CHECK_CUDA(cudaMemset(dE,0, sizeof(double)));
    dim3 block(256), grid((N+block.x-1)/block.x);
    size_t sh = 2*block.x*sizeof(double);
    reduce_mass_energy<<<grid,block,sh>>>(dU,N,N?dM:nullptr,N?dE:nullptr);
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaMemcpy(&M,dM,sizeof(double),cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(&E,dE,sizeof(double),cudaMemcpyDeviceToHost));
    cudaFree(dM); cudaFree(dE);
}

// ---------- init ----------
void init_orszag_tang(std::vector<float>& hU, int NX, int NY, float Lx, float Ly){
    auto at = [&](int c,int x,int y)->float& { return hU[c*NX*NY + y*NX + x]; };
    for (int y=0;y<NY;y++){
        for (int x=0;x<NX;x++){
            float X = (x + 0.5f) * Lx / NX;
            float Y = (y + 0.5f) * Ly / NY;

            float rho=1.0f;
            float vx =-sinf(2.f*M_PI*Y);
            float vy = sinf(2.f*M_PI*X);
            float bx =-sinf(2.f*M_PI*Y);
            float by = sinf(4.f*M_PI*X);
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

// ---------- main ----------
int main(int argc,char** argv){
    int NX    = (argc>1)? atoi(argv[1]) : 256;
    int NY    = (argc>2)? atoi(argv[2]) : 256;
    int STEPS = (argc>3)? atoi(argv[3]) : 400;
    int log_every = 50;
    float theta = 1.0f;   // minmod; try 1.5 for MC-ish limiter

    float Lx=1.0f, Ly=1.0f, dx=Lx/NX, dy=Ly/NY;

    size_t cells = (size_t)NX*NY, bytes = 6*cells*sizeof(float);
    std::vector<float> hU(6*cells), hUout(6*cells);
    init_orszag_tang(hU,NX,NY,Lx,Ly);

    float *dU=nullptr,*dV=nullptr;
    CHECK_CUDA(cudaMalloc(&dU,bytes));
    CHECK_CUDA(cudaMalloc(&dV,bytes));
    CHECK_CUDA(cudaMemcpy(dU,hU.data(),bytes,cudaMemcpyHostToDevice));

    dim3 block(16,16), grid((NX+block.x-1)/block.x,(NY+block.y-1)/block.y);

    double M0=0.0, E0=0.0;
    compute_invariants_gpu(dU,NX,NY,M0,E0);
    printf("Init invariants: Mass=%.8e  Energy=%.8e  (USE_HLL=%d, USE_POWELL=%d)\n",
           M0,E0,(int)USE_HLL,(int)USE_POWELL);

    for (int s=1; s<=STEPS; ++s){
        float dt = compute_dt_gpu(dU,NX,NY,dx,dy);
        step_mhd_muscl_powell<<<grid,block>>>(dU,dV,NX,NY,dx,dy,dt,theta);
        CHECK_CUDA(cudaPeekAtLastError());
        std::swap(dU,dV);

        if (s % log_every == 0){
            double M=0.0,E=0.0;
            compute_invariants_gpu(dU,NX,NY,M,E);
            printf("Step %4d/%4d  dt=%.3e  Mass=%.8e  dM=%.3e  Energy=%.8e  dE=%.3e\n",
                   s,STEPS,dt,M, (M-M0)/M0, E,(E-E0)/E0);
        }
    }

    CHECK_CUDA(cudaMemcpy(hUout.data(), dU, bytes, cudaMemcpyDeviceToHost));
    auto atH=[&](int c,int x,int y)->float{ return hUout[c*NX*NY + y*NX + x]; };
    int cx=NX/2, cy=NY/2;
    printf("Center (rho,mx,my,Bx,By,E): %.6f %.6f %.6f %.6f %.6f %.6f\n",
           atH(0,cx,cy), atH(1,cx,cy), atH(2,cx,cy), atH(3,cx,cy), atH(4,cx,cy), atH(5,cx,cy));

    cudaFree(dU); cudaFree(dV);
    return 0;
}
