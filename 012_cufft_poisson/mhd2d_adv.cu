#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>
#include <algorithm>
#include <filesystem>
#include "poisson_fft.cuh"

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
    F[3] = 0.0f;                // see CT note
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

// ---------- Riemann: Rusanov ----------
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
    float PL[6], float PC[6], float PR[6], // primitives at L,C,R cells along a line
    float theta,                           // limiter parameter (1.0 = minmod)
    float P_L_on_face[6], float P_R_on_face[6])
{
    // Compute limited slope (minmod of forward/backward/central * theta)
    #pragma unroll
    for (int k=0;k<6;k++){
        float dl = PC[k]-PL[k];
        float dr = PR[k]-PC[k];
        float dc = 0.5f*(PR[k]-PL[k]);
        float s  = minmod3(dl, dr, theta*dc);
        P_L_on_face[k] = PC[k] + 0.5f*s;  // right of current cell
        P_R_on_face[k] = PC[k] - 0.5f*s;  // left of right cell
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

    // Gather 3x1 stencils in x at (i-1,i,i+1) for y=iy
    int im1 = wrap(ix-1,NX), ip1 = wrap(ix+1,NX);
    int jm1 = wrap(iy-1,NY), jp1 = wrap(iy+1,NY);

    float ULc[6], UCc[6], URc[6], DLc[6], DCc[6], URy[6]; // cons
    #pragma unroll
    for(int c=0;c<6;c++){
        ULc[c]=V.at(c,im1,iy); UCc[c]=V.at(c,ix,iy); URc[c]=V.at(c,ip1,iy);
        DLc[c]=V.at(c,ix,jm1); DCc[c]=V.at(c,ix,iy); URy[c]=V.at(c,ix,jp1);
    }

    // Convert to primitives
    float PL[6], PC[6], PR[6], PDL[6], PDC[6], PUR[6];
    cons_to_prim(ULc,PL); cons_to_prim(UCc,PC); cons_to_prim(URc,PR);
    cons_to_prim(DLc,PDL); cons_to_prim(DCc,PDC); cons_to_prim(URy,PUR);

    // Reconstruct face states (x- and y-)
    float P_LR_xL[6], P_LR_xR[6], P_LR_yD[6], P_LR_yU[6];
    recon_face_1D_prim(PL,PC,PR, theta, P_LR_xR/*C right*/, P_LR_xL/*R left*/);
    recon_face_1D_prim(PDL,PDC,PUR,theta, P_LR_yU/*C up*/,   P_LR_yD/*U down*/);

    // Convert reconstructed primitives to conservative for Riemann
    float ULx[6], URx[6], ULy[6], URy_c[6];
    prim_to_cons(P_LR_xR, ULx);   // left state at i+1/2 from cell i
    prim_to_cons(P_LR_xL, URx);   // right state at i+1/2 from cell i+1

    prim_to_cons(P_LR_yU, ULy);   // lower state at j+1/2 from cell j
    prim_to_cons(P_LR_yD, URy_c); // upper state at j+1/2 from cell j+1

    // Godunov fluxes (Rusanov)
    float FxL[6], FxR[6], GyD[6], GyU[6];

    // x faces: i-1/2 uses neighbor’s right & center’s left.
    // Reconstruct again for i-1/2: shift PL,PC,PR → (i-2,i-1,i)
    // For simplicity, reuse symmetrical construction by swapping roles:
    {
        // left face states around (i-1/2): take center=PL, right=PC, left=(i-2) approximated via PL- (PR-PC)
        float Pm2[6];  // crude extrapolation for minimal code; acceptable for starter
        #pragma unroll
        for(int k=0;k<6;k++) Pm2[k] = PL[k] - (PR[k]-PC[k]); // mirror
        float PLfaceL[6], PLfaceR[6];
        recon_face_1D_prim(Pm2,PL,PC,theta, PLfaceR, PLfaceL);
        float ULxL[6], URxL[6];
        prim_to_cons(PLfaceR, ULxL); prim_to_cons(PLfaceL, URxL);
        rusanov_flux(ULxL, URxL, 0, FxL);
    }
    // right face i+1/2:
    rusanov_flux(ULx, URx, 0, FxR);

    // y faces:
    {
        // bottom face j-1/2 similar trick
        float Pjm2[6];
        #pragma unroll
        for(int k=0;k<6;k++) Pjm2[k] = PDL[k] - (PUR[k]-PDC[k]);
        float PDfaceU[6], PDfaceD[6];
        recon_face_1D_prim(Pjm2, PDL, PDC, theta, PDfaceU, PDfaceD);
        float ULyB[6], URyB[6];
        prim_to_cons(PDfaceU, ULyB); prim_to_cons(PDfaceD, URyB);
        rusanov_flux(ULyB, URyB, 1, GyD);
    }
    rusanov_flux(ULy, URy_c, 1, GyU);

    // FV update
    float Uc[6];
    #pragma unroll
    for(int c=0;c<6;c++) Uc[c]=UCc[c];

    float dUx[6], dUy[6], Un[6];
    #pragma unroll
    for(int c=0;c<6;c++){
        dUx[c] = -(FxR[c] - FxL[c]) / dx;
        dUy[c] = -(GyU[c] - GyD[c]) / dy;
        Un[c]  = Uc[c] + dt*(dUx[c] + dUy[c]);
    }

#if USE_POWELL
    // Powell 8-wave cleaning: add source ~ -(∇·B)*[0, Bx, By, vx, vy, v·B]
    // Compute discrete divB at cell center via central diffs
    float bx_im1 = V.at(3,wrap(ix-1,NX),iy), bx_ip1 = V.at(3,wrap(ix+1,NX),iy);
    float by_jm1 = V.at(4,ix,wrap(iy-1,NY)), by_jp1 = V.at(4,ix,wrap(iy+1,NY));
    float divB = (bx_ip1 - bx_im1)/(2.f*dx) + (by_jp1 - by_jm1)/(2.f*dy);

    float r = safe_rho(Un[0]);
    float vx = Un[1]/r, vy = Un[2]/r;
    float vb = vx*Un[3] + vy*Un[4];

    Un[1] += -dt * divB * Un[3];   // mx   -= dt * divB * Bx
    Un[2] += -dt * divB * Un[4];   // my   -= dt * divB * By
    Un[3] += -dt * divB * vx;      // Bx   -= dt * divB * vx
    Un[4] += -dt * divB * vy;      // By   -= dt * divB * vy
    Un[5] += -dt * divB * vb;      // E    -= dt * divB * v·B
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

    // store
    int idx = iy*NX + ix;
    #pragma unroll
    for(int c=0;c<6;c++) Uout[c*NX*NY + idx] = Un[c];
}

// ---------- max wavespeed (CFL) ----------
__device__ unsigned int d_max_speed_bits;

__global__ void reset_maxspeed(){ d_max_speed_bits = 0u; }

__device__ __forceinline__ void atomicMaxFloatPos(float v){
    // assume v>=0; reinterp to uint preserves order for non-negatives
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
    // one block -> two partials via shared mem
    extern __shared__ double sm[];
    double* s_mass = sm;
    double* s_energy = sm + blockDim.x;

    double mass = 0.0, energy = 0.0;
    for (int i=blockIdx.x*blockDim.x + threadIdx.x; i<N; i+=gridDim.x*blockDim.x){
        mass   += (double)U[i]; // rho plane stored first
        energy += (double)U[5*N + i];
    }
    s_mass[threadIdx.x]   = mass;
    s_energy[threadIdx.x] = energy;
    __syncthreads();

    // reduction
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
    reduce_mass_energy<<<grid,block,sh>>>(dU,N,dM,dE);
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
    float theta = 1.0f;   // minmod; 1.5 = MC-ish

    float Lx=1.0f, Ly=1.0f, dx=Lx/NX, dy=Ly/NY;

    size_t cells = (size_t)NX*NY;
    size_t bytes = 6*cells*sizeof(float);
    std::vector<float> hU(6*cells), hUout(6*cells);
    init_orszag_tang(hU,NX,NY,Lx,Ly);

    float *dU=nullptr,*dV=nullptr,*d_phi=nullptr;
    CHECK_CUDA(cudaMalloc(&dU,bytes));
    CHECK_CUDA(cudaMalloc(&dV,bytes));
    CHECK_CUDA(cudaMemcpy(dU,hU.data(),bytes,cudaMemcpyHostToDevice));

    dim3 block(16,16), grid((NX+block.x-1)/block.x,(NY+block.y-1)/block.y);

    Poisson2D poisson; poisson.init(NX,NY);
    Spectrum2D spec;   spec.init(NX,NY);
    CHECK_CUDA(cudaMalloc(&d_phi, cells*sizeof(float)));

    double M0=0.0, E0=0.0;
    compute_invariants_gpu(dU,NX,NY,M0,E0);
    printf("Init invariants: Mass=%.8e  Energy=%.8e\n", M0, E0);

    static std::vector<float> h_snap;
    static bool frames_ready=false;

    for (int s=1; s<=STEPS; ++s){
        float dt = compute_dt_gpu(dU,NX,NY,dx,dy);
        step_mhd_muscl_powell<<<grid,block>>>(dU,dV,NX,NY,dx,dy,dt,theta);
        CHECK_CUDA(cudaPeekAtLastError());
        std::swap(dU,dV);

        // Poisson solve on rho (mean removed)
        std::vector<float> tmpRho(cells);
        CHECK_CUDA(cudaMemcpy(tmpRho.data(), dU, cells*sizeof(float), cudaMemcpyDeviceToHost));
        double sum=0.0;
        for(size_t i=0;i<cells;i++) sum += tmpRho[i];
        float mean_rho = float(sum / double(cells));
        poisson.solve(/*rhs=*/dU, /*phi=*/d_phi, mean_rho, dx, dy);

    #if APPLY_POISSON_GRAD
        dim3 b2(16,16), g2((NX+15)/16,(NY+15)/16);
        apply_grad_phi<<<g2,b2>>>(/*mx*/dU + 1*cells, /*my*/dU + 2*cells,
                                  d_phi, NX, NY, dx, dy, dt);
        CHECK_CUDA(cudaDeviceSynchronize());
    #endif

        if (s % log_every == 0){
            double M=0.0,E=0.0;
            compute_invariants_gpu(dU,NX,NY,M,E);
            printf("Step %4d/%4d  dt=%.3e  Mass=%.8e  dM=%.3e  Energy=%.8e  dE=%.3e",
                   s,STEPS,dt,M, (M-M0)/M0, E,(E-E0)/E0);
            auto H = spec.compute(/*rho*/dU + 0*cells, /*mx*/dU + 1*cells, /*my*/dU + 2*cells, /*nbins=*/64);
            printf("  spectrum[0..7]:");
            for(int i=0;i<8 && i<(int)H.size();i++) printf(" %.3e", H[i]);
            printf("\n");

            // dump frames for animation (rho and phi)
            if (!frames_ready){
                std::filesystem::create_directories("frames");
                frames_ready=true;
            }
            if (h_snap.size() != cells) h_snap.resize(cells);
            auto dump_field = [&](const float* dptr, const char* tag){
                CHECK_CUDA(cudaMemcpy(h_snap.data(), dptr, cells*sizeof(float), cudaMemcpyDeviceToHost));
                std::string fname = "frames/" + std::string(tag) + "_" + std::to_string(s) + ".bin";
                FILE* fp = fopen(fname.c_str(),"wb");
                if (fp){ fwrite(h_snap.data(), sizeof(float), cells, fp); fclose(fp); }
            };
            dump_field(dU + 0*cells, "rho");
            dump_field(d_phi,        "phi");
        }
    }

    poisson.destroy(); spec.destroy();
    cudaFree(d_phi);

    // snapshot center
    CHECK_CUDA(cudaMemcpy(hUout.data(), dU, bytes, cudaMemcpyDeviceToHost));
    auto atH=[&](int c,int x,int y)->float{ return hUout[c*NX*NY + y*NX + x]; };
    int cx=NX/2, cy=NY/2;
    printf("Center (rho,mx,my,Bx,By,E): %.6f %.6f %.6f %.6f %.6f %.6f\n",
           atH(0,cx,cy), atH(1,cx,cy), atH(2,cx,cy), atH(3,cx,cy), atH(4,cx,cy), atH(5,cx,cy));

    cudaFree(dU); cudaFree(dV);
    return 0;
}
