#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <vector>

static void check(cudaError_t status, const char *msg) {
    if (status != cudaSuccess) {
        std::fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(status));
        std::exit(1);
    }
}

static void run_alloc(bool use_pool) {
    if (use_pool) {
        cudaMemPoolProps props{};
        props.allocType = cudaMemAllocationTypePinned;
        props.location.type = cudaMemLocationTypeDevice;
        props.location.id = 0;
        cudaMemPool_t pool;
        check(cudaMemPoolCreate(&pool, &props), "create pool");
        check(cudaDeviceSetMemPool(0, pool), "set pool");
    }

    std::vector<void *> ptrs;
    for (int i = 0; i < 100; ++i) {
        size_t size = (i % 10 + 1) * 1024 * 1024;
        void *ptr = nullptr;
        check(cudaMallocAsync(&ptr, size, 0), "malloc async");
        ptrs.push_back(ptr);
        if (i % 3 == 0 && !ptrs.empty()) {
            check(cudaFreeAsync(ptrs.back(), 0), "free async");
            ptrs.pop_back();
        }
    }
    for (void *ptr : ptrs) {
        check(cudaFreeAsync(ptr, 0), "free async");
    }
    check(cudaDeviceSynchronize(), "sync");

    if (use_pool) {
        cudaMemPool_t pool;
        check(cudaDeviceGetMemPool(&pool, 0), "get pool");
        check(cudaMemPoolDestroy(pool), "destroy pool");
    }
}

int main() {
    run_alloc(false);
    run_alloc(true);
    std::printf("fragmentation test done\n");
    return 0;
}
