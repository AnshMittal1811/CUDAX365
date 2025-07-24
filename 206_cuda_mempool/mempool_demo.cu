#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

static void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

int main() {
    cudaMemPool_t pool;
    check(cudaDeviceGetDefaultMemPool(&pool, 0), "get default mempool");

    cudaMemPoolProps props{};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.type = cudaMemLocationTypeDevice;
    props.location.id = 0;

    cudaMemPool_t custom_pool;
    check(cudaMemPoolCreate(&custom_pool, &props), "create mempool");
    check(cudaDeviceSetMemPool(0, custom_pool), "set mempool");

    void *ptr = nullptr;
    check(cudaMallocAsync(&ptr, 64 * 1024 * 1024, 0), "cudaMallocAsync");
    check(cudaFreeAsync(ptr, 0), "cudaFreeAsync");
    check(cudaDeviceSynchronize(), "sync");

    std::printf("mempool allocation done\n");

    check(cudaDeviceSetMemPool(0, pool), "restore pool");
    check(cudaMemPoolDestroy(custom_pool), "destroy mempool");

    return 0;
}
