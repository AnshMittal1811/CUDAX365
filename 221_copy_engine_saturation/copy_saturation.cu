#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void compute_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 1.001f + 0.1f;
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 22);
    int streams = (argc > 2) ? std::atoi(argv[2]) : 4;

    size_t bytes = n * sizeof(float);
    float *h = nullptr;
    cudaHostAlloc(&h, bytes, cudaHostAllocDefault);

    float *d = nullptr;
    cudaMalloc(&d, bytes);

    cudaStream_t *stream = new cudaStream_t[streams];
    for (int i = 0; i < streams; ++i) {
        cudaStreamCreate(&stream[i]);
    }

    int chunk = n / streams;
    for (int i = 0; i < streams; ++i) {
        size_t offset = static_cast<size_t>(i) * chunk;
        cudaMemcpyAsync(d + offset, h + offset, chunk * sizeof(float), cudaMemcpyHostToDevice, stream[i]);
        compute_kernel<<<(chunk + 255) / 256, 256, 0, stream[i]>>>(d + offset, chunk);
        cudaMemcpyAsync(h + offset, d + offset, chunk * sizeof(float), cudaMemcpyDeviceToHost, stream[i]);
    }

    for (int i = 0; i < streams; ++i) {
        cudaStreamSynchronize(stream[i]);
        cudaStreamDestroy(stream[i]);
    }

    cudaFree(d);
    cudaFreeHost(h);
    delete[] stream;

    std::printf("copy saturation done\n");
    return 0;
}
