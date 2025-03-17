#include <curand_kernel.h>
#include <cstdio>

__global__ void init_sobol(curandStateScrambledSobol64* states, int n, unsigned long long seed){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &states[i]);
}

__global__ void sobol_kernel(curandStateScrambledSobol64* states, float* out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStateScrambledSobol64 local = states[i];
    unsigned long long v = curand(&local);
    out[i] = (v & 0xFFFFFF) / float(0xFFFFFF);
    states[i] = local;
}

int main(){
    int n = 256;
    curandStateScrambledSobol64* states = nullptr;
    float* out = nullptr;
    cudaMalloc(&states, n * sizeof(curandStateScrambledSobol64));
    cudaMalloc(&out, n * sizeof(float));
    init_sobol<<<1, 256>>>(states, n, 1234ULL);
    sobol_kernel<<<1, 256>>>(states, out, n);
    cudaDeviceSynchronize();
    printf("sobol done\n");

    cudaFree(states);
    cudaFree(out);
    return 0;
}
