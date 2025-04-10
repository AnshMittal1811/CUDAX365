#include <cuda_runtime.h>
#include <cstdio>
#include <cmath>

__device__ __forceinline__ float add_rn(float a, float b){
    float out;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ unsigned lcg(unsigned &state){
    state = state * 1664525u + 1013904223u;
    return state;
}

__device__ __forceinline__ float rand01(unsigned &state){
    return (lcg(state) & 0xFFFFFF) / float(0xFFFFFF);
}

__device__ __forceinline__ float cost_fn(float x){
    // Simple convex function; minimum near x=0.3
    float d = x - 0.3f;
    return d * d + 0.05f * sinf(10.0f * x);
}


__global__ void anneal_kernel(float* candidates, float* costs, int steps, float t0){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned state = 1234u + i * 1663u;
    float x = candidates[i];
    float best = cost_fn(x);

    for (int s = 0; s < steps; ++s){
        float t = t0 * (1.0f - s / float(steps));
        float delta = (rand01(state) - 0.5f) * 0.1f;
        float xn = x + delta;
        float c = cost_fn(xn);
        float dc = c - best;
        float accept = expf(-dc / fmaxf(t, 1e-6f));
        if (dc < 0.0f || rand01(state) < accept){
            x = xn;
            best = c;
        }
    }
    candidates[i] = x;
    costs[i] = best;
}

int main(){
    int n = 1024;
    int steps = 200;
    float t0 = 0.2f;
    float *d_cand = nullptr, *d_cost = nullptr;
    cudaMalloc(&d_cand, n * sizeof(float));
    cudaMalloc(&d_cost, n * sizeof(float));

    // init candidates
    float* h = new float[n];
    for (int i=0;i<n;i++) h[i] = (i % 100) / 100.0f;
    cudaMemcpy(d_cand, h, n * sizeof(float), cudaMemcpyHostToDevice);

    int block = 256;
    int grid = (n + block - 1) / block;
    anneal_kernel<<<grid, block>>>(d_cand, d_cost, steps, t0);
    cudaDeviceSynchronize();

    cudaMemcpy(h, d_cost, n * sizeof(float), cudaMemcpyDeviceToHost);
    float best = h[0];
    for (int i=1;i<n;i++) if (h[i] < best) best = h[i];
    printf("best_cost=%.6f\n", best);

    cudaFree(d_cand);
    cudaFree(d_cost);
    delete[] h;
    return 0;
}
