#include <cstdio>
#include <cmath>
#include <vector>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstring>

static inline float uint_as_float_host(uint32_t u) {
    float f;
    std::memcpy(&f, &u, sizeof(float));
    return f;
}

#define CHECK_CUDA(x) do { auto err=(x); if (err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)

constexpr float GAMMA   = 1.4f;
constexpr float RHO_MIN = 1e-6f;
constexpr float P_MIN   = 1e-6f;
constexpr float CFL     = 0.30f;

__device__ __forceinline__ float safe_rho(float rho){ return fmaxf(rho, RHO_MIN); }

__device__ __forceinline__ float pressure(float rho, float mx, float my, float bx, float by, float E){
    float r  = safe_rho(rho);
    float vx = mx / r;
    float vy = my / r;
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
    float p = pressure(rho,mx,my,bx,by,E);
    float bt2 = bx*bx + by*by;
    float pt = p + 0.5f*bt2;
    float vb = vx*bx + vy*by;

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
    float p = pressure(rho,mx,my,bx,by,E);
    float bt2 = bx*bx + by*by;
    float pt = p + 0.5f*bt2;
    float vb = vx*bx + vy*by;

    G[0] = my;
    G[1] = my*vx - by*bx;
    G[2] = my*vy + pt - by*by;
    G[3] = vx*by - vy*bx;
    G[4] = 0.0f;
    G[5] = (E + pt)*vy - by*vb;
}

__device__ __forceinline__ void rusanov_flux(
    const float UL[6], const float UR[6],
    int dirXY, float outF[6])
{
    float FL[6], FR[6];
    if (dirXY==0) { flux_x(UL, FL); flux_x(UR, FR); }
    else          { flux_y(UL, FL); flux_y(UR, FR); }

    auto vn = [&](const float U[6]){
        float r=safe_rho(U[0]);
        float vx=U[1]/r, vy=U[2]/r;
        return (dirXY==0) ? vx : vy;
    };
    float pL = pressure(UL[0],UL[1],UL[2],UL[3],UL[4],UL[5]);
    float pR = pressure(UR[0],UR[1],UR[2],UR[3],UR[4],UR[5]);
    float cL = fabsf(vn(UL)) + fast_magnetosonic(UL[0], pL, UL[3], UL[4]);
    float cR = fabsf(vn(UR)) + fast_magnetosonic(UR[0], pR, UR[3], UR[4]);
    float smax = fmaxf(cL, cR);

    #pragma unroll
    for (int m=0;m<6;m++)
        outF[m] = 0.5f*(FL[m] + FR[m]) - 0.5f*smax*(UR[m] - UL[m]);
}

__device__ __forceinline__ int wrap(int i, int n){ i%=n; if(i<0) i+=n; return i; }

__global__ void step_mhd(
    const float* __restrict__ Uin,
    float* __restrict__ Uout,
    int NX, int NY, float dx, float dy, float dt)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=NX || iy>=NY) return;

    auto at = [&](int comp, int x, int y)->float {
        int idx = comp*NX*NY + y*NX + x;
        return Uin[idx];
    };
    auto st = [&](int comp, int x, int y, float v){
        int idx = comp*NX*NY + y*NX + x;
        Uout[idx] = v;
    };

    float UL[6], UR[6], Uc[6];
    int L = wrap(ix-1, NX), R = wrap(ix+1, NX);
    int D = wrap(iy-1, NY), U = wrap(iy+1, NY);

    #pragma unroll
    for (int c=0;c<6;c++) Uc[c] = at(c, ix, iy);

    #pragma unroll
    for (int c=0;c<6;c++){ UL[c]=at(c, L, iy); UR[c]=at(c, R, iy); }
    float FxL[6], FxR[6];
    rusanov_flux(UL, Uc, 0, FxL);
    rusanov_flux(Uc, UR, 0, FxR);

    #pragma unroll
    for (int c=0;c<6;c++){ UL[c]=at(c, ix, D); UR[c]=at(c, ix, U); }
    float GyD[6], GyU[6];
    rusanov_flux(UL, Uc, 1, GyD);
    rusanov_flux(Uc, UR, 1, GyU);

    #pragma unroll
    for (int c=0;c<6;c++){
        float dUx = -(FxR[c] - FxL[c]) / dx;
        float dUy = -(GyU[c] - GyD[c]) / dy;
        float Un  = Uc[c] + dt * (dUx + dUy);

        // floor to keep state valid during early experiments
        if (c==0) Un = fmaxf(Un, RHO_MIN); // rho
        if (c==5){
            float r = safe_rho(Un); // tentative
            (void)r;
        }
        st(c, ix, iy, Un);
    }

    // Fix energy floor after we updated everything
    float rho = Uout[0*NX*NY + iy*NX + ix];
    float mx  = Uout[1*NX*NY + iy*NX + ix];
    float my  = Uout[2*NX*NY + iy*NX + ix];
    float bx  = Uout[3*NX*NY + iy*NX + ix];
    float by  = Uout[4*NX*NY + iy*NX + ix];
    float E   = Uout[5*NX*NY + iy*NX + ix];

    // Ensure positive pressure
    float p    = pressure(rho,mx,my,bx,by,E);
    float r    = safe_rho(rho);
    float vx   = mx/r, vy=my/r;
    float kin  = 0.5f * r * (vx*vx + vy*vy);
    float mag  = 0.5f * (bx*bx + by*by);
    float Emin = kin + mag + P_MIN/(GAMMA-1.f);
    if (E < Emin) E = Emin;
    Uout[5*NX*NY + iy*NX + ix] = E;
}

// ---- Max wavespeed (for CFL dt) -------------------------------------------
__device__ unsigned int d_max_speed_bits;

__global__ void reset_maxspeed(){ d_max_speed_bits = 0; }

// reinterpret positive float as uint for atomicMax
__device__ __forceinline__ void atomicMaxFloat(float v){
    unsigned int ui = __float_as_uint(v);
    atomicMax(&d_max_speed_bits, ui);
}

__global__ void kernel_maxspeed(
    const float* __restrict__ U, int NX, int NY)
{
    int ix = blockIdx.x*blockDim.x + threadIdx.x;
    int iy = blockIdx.y*blockDim.y + threadIdx.y;
    if (ix>=NX || iy>=NY) return;

    int idx = iy*NX + ix;
    float rho = U[0*NX*NY + idx];
    float mx  = U[1*NX*NY + idx];
    float my  = U[2*NX*NY + idx];
    float bx  = U[3*NX*NY + idx];
    float by  = U[4*NX*NY + idx];
    float E   = U[5*NX*NY + idx];

    float r  = safe_rho(rho);
    float vx = mx/r, vy=my/r;
    float p  = pressure(rho,mx,my,bx,by,E);
    float cf = fast_magnetosonic(rho,p,bx,by);

    float smax = fmaxf(fabsf(vx)+cf, fabsf(vy)+cf);
    atomicMaxFloat(smax);
}

float compute_dt_gpu(const float* dU, int NX, int NY, float dx, float dy){
    dim3 block(16,16);
    dim3 grid((NX+block.x-1)/block.x, (NY+block.y-1)/block.y);
    reset_maxspeed<<<1,1>>>();
    kernel_maxspeed<<<grid,block>>>(dU, NX, NY);
    CHECK_CUDA(cudaDeviceSynchronize());

    unsigned int hbits = 0;
    CHECK_CUDA(cudaMemcpyFromSymbol(&hbits, d_max_speed_bits,
                                    sizeof(unsigned int), 0,
                                    cudaMemcpyDeviceToHost));
    float smax = uint_as_float_host(hbits);   // <-- use host helper
    if (!std::isfinite(smax) || smax <= 0.f) smax = 1.0f;

    float dt = CFL * fminf(dx, dy) / smax;
    return dt;
}

// ---- Init ------------------------------------------------------------------
void init_orszag_tang(std::vector<float>& hU, int NX, int NY, float Lx, float Ly){
    auto at = [&](int c, int x, int y)->float& { return hU[c*NX*NY + y*NX + x]; };
    for (int y=0;y<NY;y++){
        for (int x=0;x<NX;x++){
            float X = (x + 0.5f) * Lx / NX;
            float Y = (y + 0.5f) * Ly / NY;

            float rho = 1.0f;
            float vx = -sinf(2.f*M_PI*Y);
            float vy =  sinf(2.f*M_PI*X);
            float bx = -sinf(2.f*M_PI*Y);
            float by =  sinf(4.f*M_PI*X);
            float p  =  1.0f;

            float mx = rho * vx;
            float my = rho * vy;
            float kin = 0.5f * rho * (vx*vx + vy*vy);
            float mag = 0.5f * (bx*bx + by*by);
            float E   = p/(GAMMA-1.f) + kin + mag;

            at(0,x,y)=rho; at(1,x,y)=mx; at(2,x,y)=my;
            at(3,x,y)=bx;  at(4,x,y)=by; at(5,x,y)=E;
        }
    }
}

int main(int argc, char** argv){
    int NX = (argc>1)? atoi(argv[1]) : 256;
    int NY = (argc>2)? atoi(argv[2]) : 256;
    int STEPS = (argc>3)? atoi(argv[3]) : 200;

    float Lx=1.0f, Ly=1.0f;
    float dx=Lx/NX, dy=Ly/NY;

    size_t cells = (size_t)NX*NY;
    size_t bytes = 6 * cells * sizeof(float);

    std::vector<float> hU(6*cells), hUout(6*cells);
    init_orszag_tang(hU, NX, NY, Lx, Ly);

    float *dU=nullptr, *dV=nullptr;
    CHECK_CUDA(cudaMalloc(&dU, bytes));
    CHECK_CUDA(cudaMalloc(&dV, bytes));
    CHECK_CUDA(cudaMemcpy(dU, hU.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(16,16);
    dim3 grid( (NX+block.x-1)/block.x, (NY+block.y-1)/block.y );

    for (int s=0; s<STEPS; ++s){
        float dt = compute_dt_gpu(dU, NX, NY, dx, dy);
        step_mhd<<<grid, block>>>(dU, dV, NX, NY, dx, dy, dt);
        CHECK_CUDA(cudaPeekAtLastError());
        std::swap(dU, dV);
        if ((s%50)==0) { printf("Step %d/%d (dt=%.3e)\n", s, STEPS, dt); }
    }
    if (STEPS%2!=0) std::swap(dU, dV);

    CHECK_CUDA(cudaMemcpy(hUout.data(), dV, bytes, cudaMemcpyDeviceToHost));

    auto atH = [&](int c, int x, int y)->float { return hUout[c*NX*NY + y*NX + x]; };
    int cx=NX/2, cy=NY/2;
    printf("Center (rho,mx,my,Bx,By,E): %.6f %.6f %.6f %.6f %.6f %.6f\n",
           atH(0,cx,cy), atH(1,cx,cy), atH(2,cx,cy), atH(3,cx,cy), atH(4,cx,cy), atH(5,cx,cy));

    cudaFree(dU); cudaFree(dV);
    return 0;
}
