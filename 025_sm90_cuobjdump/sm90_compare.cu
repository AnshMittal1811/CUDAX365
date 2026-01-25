#include <cuda_runtime.h>
#include <cstdio>
#include <vector>

#define CUDA_CHECK(call) do { \
    cudaError_t err__ = (call); \
    if (err__ != cudaSuccess) { \
        std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
        return 1; \
    } \
} while (0)

__global__ void saxpy(const float* x, const float* y, float* out, float a, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = a * x[i] + y[i];
}

int main(){
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    CUDA_CHECK(cudaSetDevice(device));

    int sm = prop.major * 10 + prop.minor;
    int n = (sm >= 89) ? (1 << 24) : (1 << 20);
    size_t bytes = n * sizeof(float);
    std::vector<float> hx(n, 1.0f), hy(n, 2.0f), hout(n, 0.0f);
    float *dx = nullptr, *dy = nullptr, *dout = nullptr;

    std::printf("device=%s compute=%d.%d global_mem_mib=%zu\n",
                prop.name, prop.major, prop.minor,
                static_cast<size_t>(prop.totalGlobalMem / (1024 * 1024)));
    std::printf("native target for RTX 4090 Laptop GPU is sm_89; elements=%d bytes_per_vector=%zu\n", n, bytes);

    CUDA_CHECK(cudaMalloc(&dx, bytes));
    CUDA_CHECK(cudaMalloc(&dy, bytes));
    CUDA_CHECK(cudaMalloc(&dout, bytes));
    CUDA_CHECK(cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice));
    int block = 256;
    int grid = (n + block - 1) / block;
    saxpy<<<grid, block>>>(dx, dy, dout, 2.5f, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(hout.data(), dout, bytes, cudaMemcpyDeviceToHost));
    std::printf("out[0]=%.3f out[last]=%.3f expected=4.500\n", hout[0], hout[n - 1]);
    CUDA_CHECK(cudaFree(dx));
    CUDA_CHECK(cudaFree(dy));
    CUDA_CHECK(cudaFree(dout));
    return 0;
}
