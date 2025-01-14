#include <cuda_runtime.h>
#include <cufft.h>
#include <cstdio>
#include <vector>
#include <cmath>
#include <cstdlib>

#ifndef CHECK_CUDA
#define CHECK_CUDA(x) do { auto err=(x); if(err!=cudaSuccess){ \
  fprintf(stderr,"CUDA error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(err)); exit(1);} } while(0)
#endif

#ifndef CHECK_CUFFT
#define CHECK_CUFFT(x) do { cufftResult _e=(x); if(_e!=CUFFT_SUCCESS){ \
  fprintf(stderr,"CUFFT error %s:%d: code=%d\n", __FILE__,__LINE__,(int)_e); exit(1);} } while(0)
#endif

#ifndef CHECK_LAST
#define CHECK_LAST() do { auto _e=cudaPeekAtLastError(); if(_e!=cudaSuccess){ \
  fprintf(stderr,"Kernel launch error %s:%d: %s\n", __FILE__,__LINE__,cudaGetErrorString(_e)); exit(1);} } while(0)
#endif

// ---------- helpers ----------
__device__ __forceinline__ int lane_id()  { return threadIdx.x & 31; }
__device__ __forceinline__ int warp_id()  { return threadIdx.x >> 5; }
__device__ __forceinline__ int brev5(int x){ return __brev(x) >> 27; } // reverse 5 bits

// inline PTX complex multiply: (a.x + i a.y) * (b.x + i b.y)
__device__ __forceinline__ float2 cmul_ptx(const float2 a, const float2 b){
    float rx, ry;
    asm volatile(
      "{\n\t"
      "mul.f32 %0, %2, %4;\n\t"          // a.x*b.x
      "mul.f32 %1, %3, %5;\n\t"          // a.y*b.y
      "sub.f32 %0, %0, %1;\n\t"          // rx = ax*bx - ay*by
      "mul.f32 %1, %2, %5;\n\t"          // ax*by
      "fma.rn.f32 %1, %3, %4, %1;\n\t"   // ry = ax*by + ay*bx
      "}\n"
      : "=&f"(rx), "=&f"(ry)
      : "f"(a.x), "f"(a.y), "f"(b.x), "f"(b.y)
    );
    return make_float2(rx, ry);
}

// Twiddle for stage s (0..4) at lane l (0..31)
__device__ __forceinline__ float2 twiddle32(int stage, int lane){
    const int m   = 1 << (stage + 1);
    const int j   = lane & ((m >> 1) - 1);
    const float angle = -2.0f * 3.14159265358979323846f * (float)j / (float)m;
    float s = __sinf(angle);
    float c = __cosf(angle);
    return make_float2(c, s);
}

// One in-warp 32-pt DIT FFT using XOR butterflies
__device__ __forceinline__ float2 fft32_dit_inwarp(float2 v){
    const unsigned mask = 0xffffffffu;
    #pragma unroll
    for (int s=0; s<5; ++s){
        const int stride = 1 << s;
        const float2 w = twiddle32(s, lane_id());
        float2 p;
        p.x = __shfl_xor_sync(mask, v.x, stride, 32);
        p.y = __shfl_xor_sync(mask, v.y, stride, 32);
        if (lane_id() & stride){
            float2 t = cmul_ptx(v, w);
            v.x = p.x - t.x;
            v.y = p.y - t.y;
        } else {
            float2 t = cmul_ptx(p, w);
            v.x = v.x + t.x;
            v.y = v.y + t.y;
        }
    }
    // bit-reverse to natural order
    const int out_lane = brev5(lane_id());
    float2 out;
    out.x = __shfl_sync(mask, v.x, out_lane, 32);
    out.y = __shfl_sync(mask, v.y, out_lane, 32);
    return out;
}

// 32x32 tile FFT: 32 warps/block
__global__ void fft32x32_warp_ptx(const float2* __restrict__ in,
                                  float2* __restrict__ out,
                                  int tiles)
{
    if (blockIdx.x >= tiles) return;
    __shared__ float2 tile[32][33]; // +1 pad to avoid bank conflicts

    const int l  = lane_id(); // 0..31 column
    const int w  = warp_id(); // 0..31 row
    const int base = blockIdx.x * 32 * 32;

    // Row FFT
    float2 v = in[base + w*32 + l];
    v = fft32_dit_inwarp(v);
    tile[w][l] = v;
    __syncthreads();

    // Column FFT (transpose read)
    float2 vc = tile[l][w];
    vc = fft32_dit_inwarp(vc);
    out[base + l*32 + w] = vc;
}

// ------------------------------- Test & compare -------------------------------
int main(int argc, char** argv){
    int tiles = (argc>1)? atoi(argv[1]) : 1;
    const int TILE_ELEMS = 32*32;
    const int N = tiles * TILE_ELEMS;
    size_t bytes = N * sizeof(float2);

    std::vector<float2> h_in(N), h_out(N), h_ref(N);
    for (int i=0;i<N;i++){
        h_in[i].x = std::sin(0.01f*i);
        h_in[i].y = std::cos(0.02f*i);
    }

    float2 *d_in=nullptr, *d_out=nullptr, *d_ref=nullptr;
    CHECK_CUDA(cudaMalloc(&d_in, bytes));
    CHECK_CUDA(cudaMalloc(&d_out, bytes));
    CHECK_CUDA(cudaMalloc(&d_ref, bytes));
    CHECK_CUDA(cudaMemcpy(d_in, h_in.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_ref, h_in.data(), bytes, cudaMemcpyHostToDevice));

    // Our kernel
    dim3 block(1024), grid(tiles);
    fft32x32_warp_ptx<<<grid,block>>>(d_in, d_out, tiles);
    CHECK_LAST();
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s1,e1; CHECK_CUDA(cudaEventCreate(&s1)); CHECK_CUDA(cudaEventCreate(&e1));
    CHECK_CUDA(cudaEventRecord(s1));
    for(int it=0; it<2000; ++it){
        fft32x32_warp_ptx<<<grid,block>>>(d_in, d_out, tiles);
    }
    CHECK_LAST();
    CHECK_CUDA(cudaEventRecord(e1)); CHECK_CUDA(cudaEventSynchronize(e1));
    float ms1=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms1,s1,e1));

    // cuFFT reference
    cufftHandle plan;
    CHECK_CUFFT(cufftPlan2d(&plan, 32, 32, CUFFT_C2C));
    // Warmup
    for(int t=0;t<tiles;t++){
        CHECK_CUFFT(cufftExecC2C(plan,
            (cufftComplex*)(d_ref + t*TILE_ELEMS),
            (cufftComplex*)(d_ref + t*TILE_ELEMS),
            CUFFT_FORWARD));
    }
    CHECK_CUDA(cudaDeviceSynchronize());

    cudaEvent_t s2,e2; CHECK_CUDA(cudaEventCreate(&s2)); CHECK_CUDA(cudaEventCreate(&e2));
    CHECK_CUDA(cudaEventRecord(s2));
    for(int it=0; it<2000; ++it){
        for(int t=0;t<tiles;t++){
            CHECK_CUFFT(cufftExecC2C(plan,
                (cufftComplex*)(d_ref + t*TILE_ELEMS),
                (cufftComplex*)(d_ref + t*TILE_ELEMS),
                CUFFT_FORWARD));
        }
    }
    CHECK_CUDA(cudaEventRecord(e2)); CHECK_CUDA(cudaEventSynchronize(e2));
    float ms2=0.f; CHECK_CUDA(cudaEventElapsedTime(&ms2,s2,e2));

    // Compare
    CHECK_CUDA(cudaMemcpy(h_out.data(), d_out, bytes, cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_ref.data(), d_ref, bytes, cudaMemcpyDeviceToHost));

    double max_abs_err = 0.0;
    for (int i=0;i<N;i++){
        double dx = h_out[i].x - h_ref[i].x;
        double dy = h_out[i].y - h_ref[i].y;
        double e  = std::sqrt(dx*dx + dy*dy);
        if (e > max_abs_err) max_abs_err = e;
    }

    printf("PTX-FFT:  %g ms for 2000 iters (%.3f us/iter per tile)\n",
           ms1, 1000.0*ms1/2000.0/tiles);
    printf("cuFFT:    %g ms for 2000 iters (%.3f us/iter per tile)\n",
           ms2, 1000.0*ms2/2000.0/tiles);
    printf("Max |err| vs cuFFT: %.3e\n", max_abs_err);
    printf("Sample out[0]=(%f,%f)\n", h_out[0].x, h_out[0].y);

    cufftDestroy(plan);
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_ref);
    return 0;
}
