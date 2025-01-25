#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

__global__ void saxpy(const float* x, const float* y, float* out, float a, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

int main(){
    int n = 1 << 20;
    size_t bytes = n * sizeof(float);
    std::vector<float> hx(n, 1.0f), hy(n, 2.0f), hout(n, 0.0f);
    float *dx = nullptr, *dy = nullptr, *dout = nullptr;
    cudaMalloc(&dx, bytes);
    cudaMalloc(&dy, bytes);
    cudaMalloc(&dout, bytes);
    cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice);
    int block = 256;
    int grid = (n + block - 1) / block;
    saxpy<<<grid, block>>>(dx, dy, dout, 2.5f, n);
    cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyDeviceToHost);
    printf("out[0]=%.3f\n", hout[0]);
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dout);
    return 0;
}
