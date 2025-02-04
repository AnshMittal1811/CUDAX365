#include <cuda_runtime.h>
#include <cstdio>
#include <vector>
#include <cstring>

__device__ __forceinline__ unsigned int atom_min_u32(unsigned int* addr, unsigned int val){
    unsigned int old;
    asm volatile("atom.global.min.u32 %0, [%1], %2;" : "=r"(old) : "l"(addr), "r"(val) : "memory");
    return old;
}

__global__ void init_labels(const unsigned char* img, unsigned int* labels, int w, int h){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    labels[idx] = img[idx] ? idx : 0u;
}

__global__ void propagate(const unsigned char* img, unsigned int* labels, int w, int h){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    if (!img[idx]) return;

    unsigned int best = labels[idx];
    if (x > 0) best = min(best, labels[idx - 1]);
    if (x + 1 < w) best = min(best, labels[idx + 1]);
    if (y > 0) best = min(best, labels[idx - w]);
    if (y + 1 < h) best = min(best, labels[idx + w]);

    atom_min_u32(&labels[idx], best);
}

int main(){
    int w = 64, h = 64;
    int n = w * h;
    std::vector<unsigned char> h_img(n, 0);
    for (int y=16;y<48;y++) for (int x=16;x<48;x++) h_img[y*w+x] = 1;

    unsigned char* d_img = nullptr;
    unsigned int* d_labels = nullptr;
    cudaMalloc(&d_img, n);
    cudaMalloc(&d_labels, n * sizeof(unsigned int));
    cudaMemcpy(d_img, h_img.data(), n, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    init_labels<<<grid, block>>>(d_img, d_labels, w, h);
    for (int i=0;i<20;i++) propagate<<<grid, block>>>(d_img, d_labels, w, h);
    cudaDeviceSynchronize();

    std::vector<unsigned int> h_labels(n);
    cudaMemcpy(h_labels.data(), d_labels, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    printf("label at center=%u\n", h_labels[(h/2)*w + (w/2)]);

    cudaFree(d_img);
    cudaFree(d_labels);
    return 0;
}
