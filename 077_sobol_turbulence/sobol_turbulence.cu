#include <curand_kernel.h>
#include <cstdio>

__global__ void init_states(curandStateScrambledSobol64* states, int n, unsigned long long seed){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(seed, i, 0, &states[i]);
}

__global__ void add_noise(curandStateScrambledSobol64* states, float* field, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curandStateScrambledSobol64 local = states[i];
    unsigned long long v = curand(&local);
    float noise = (v & 0xFFFFFF) / float(0xFFFFFF) - 0.5f;
    field[i] += 0.01f * noise;
    states[i] = local;
}

int main(){
    int n = 256 * 256;
    float* field = nullptr;
    curandStateScrambledSobol64* states = nullptr;
    cudaMalloc(&field, n * sizeof(float));
    cudaMalloc(&states, n * sizeof(curandStateScrambledSobol64));
    cudaMemset(field, 0, n * sizeof(float));

    int block = 256;
    int grid = (n + block - 1) / block;
    init_states<<<grid, block>>>(states, n, 1234ULL);
    add_noise<<<grid, block>>>(states, field, n);
    cudaDeviceSynchronize();
    printf("turbulence noise applied\n");

    cudaFree(field);
    cudaFree(states);
    return 0;
}
