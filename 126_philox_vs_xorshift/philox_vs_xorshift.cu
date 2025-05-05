#include <curand_kernel.h>
#include <cstdio>
#include <vector>
#include <cmath>

__global__ void philox_kernel(curandStatePhilox4_32_10_t* states, float* out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    curand_init(1234ULL, i, 0, &states[i]);
    out[i] = curand_uniform(&states[i]);
}

__global__ void xorshift_kernel(unsigned int* state, float* out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned int x = state[i] ? state[i] : (i + 1) * 747796405u;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state[i] = x;
    out[i] = (x & 0xFFFFFF) / float(0xFFFFFF);
}

int main(){
    int n = 1 << 20;
    float *d_philox=nullptr, *d_xor=nullptr;
    curandStatePhilox4_32_10_t* d_states=nullptr;
    unsigned int* d_state=nullptr;

    cudaMalloc(&d_philox, n*sizeof(float));
    cudaMalloc(&d_xor, n*sizeof(float));
    cudaMalloc(&d_states, n*sizeof(curandStatePhilox4_32_10_t));
    cudaMalloc(&d_state, n*sizeof(unsigned int));
    cudaMemset(d_state, 0, n*sizeof(unsigned int));

    int block = 256;
    int grid = (n + block - 1) / block;
    philox_kernel<<<grid, block>>>(d_states, d_philox, n);
    xorshift_kernel<<<grid, block>>>(d_state, d_xor, n);
    cudaDeviceSynchronize();

    std::vector<float> h(n);
    cudaMemcpy(h.data(), d_philox, n*sizeof(float), cudaMemcpyDeviceToHost);
    double mean = 0.0;
    for (float v : h) mean += v;
    mean /= n;
    printf("philox mean=%.6f\n", mean);

    cudaMemcpy(h.data(), d_xor, n*sizeof(float), cudaMemcpyDeviceToHost);
    mean = 0.0;
    for (float v : h) mean += v;
    mean /= n;
    printf("xorshift mean=%.6f\n", mean);

    cudaFree(d_philox);
    cudaFree(d_xor);
    cudaFree(d_states);
    cudaFree(d_state);
    return 0;
}
