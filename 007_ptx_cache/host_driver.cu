#include <cuda.h>
#include <cstdio>
#include <vector>
#include <chrono>
#include <cassert>

#define CHECK_CU(x) do { CUresult r = (x); if (r != CUDA_SUCCESS) { \
  const char* errStr=nullptr; cuGetErrorString(r,&errStr); \
  fprintf(stderr,"CUDA Driver error %s:%d: %s\n", __FILE__, __LINE__, errStr?errStr:"<unknown>"); \
  std::exit(1);} } while(0)

static void run_once(const char* cubin_path, int N)
{
    // Init driver & context
    CHECK_CU(cuInit(0));
    CUdevice dev;  CHECK_CU(cuDeviceGet(&dev, 0));
    CUcontext ctx; CHECK_CU(cuCtxCreate(&ctx, 0, dev));

    // Load the assembled cubin and fetch the kernel symbol
    CUmodule mod;      CHECK_CU(cuModuleLoad(&mod, cubin_path));
    CUfunction func;   CHECK_CU(cuModuleGetFunction(&func, mod, "vec_add_kernel"));

    size_t bytes = size_t(N) * sizeof(float);

    // Host data
    std::vector<float> hA(N, 1.0f), hB(N, 2.0f), hC(N, 0.0f);

    // Device buffers
    CUdeviceptr dA, dB, dC;
    CHECK_CU(cuMemAlloc(&dA, bytes));
    CHECK_CU(cuMemAlloc(&dB, bytes));
    CHECK_CU(cuMemAlloc(&dC, bytes));

    CHECK_CU(cuMemcpyHtoD(dA, hA.data(), bytes));
    CHECK_CU(cuMemcpyHtoD(dB, hB.data(), bytes));

    // Launch params
    int block = 256;
    int grid  = (N + block - 1) / block;

    void* args[] = { &dA, &dB, &dC, &N };

    // Warm-up
    for (int i=0;i<5;i++) {
        CHECK_CU(cuLaunchKernel(func,
            grid, 1, 1,
            block, 1, 1,
            0, 0, args, nullptr));
    }
    CHECK_CU(cuCtxSynchronize());

    // Time 100 iters
    CUevent start, stop;
    CHECK_CU(cuEventCreate(&start, CU_EVENT_DEFAULT));
    CHECK_CU(cuEventCreate(&stop,  CU_EVENT_DEFAULT));

    CHECK_CU(cuEventRecord(start, 0));
    for (int it=0; it<100; ++it) {
        CHECK_CU(cuLaunchKernel(func,
            grid, 1, 1,
            block, 1, 1,
            0, 0, args, nullptr));
    }
    CHECK_CU(cuEventRecord(stop, 0));
    CHECK_CU(cuEventSynchronize(stop));

    float ms=0.f;
    CHECK_CU(cuEventElapsedTime(&ms, start, stop));

    // Copy back 1 element to sanity-check
    CHECK_CU(cuMemcpyDtoH(hC.data(), dC, sizeof(float)));
    printf("C[0]=%f (expect 3.0)\n", hC[0]);

    // Effective bandwidth (2 loads + 1 store per elem per iter)
    double bytes_per_iter = 3.0 * sizeof(float) * double(N);
    double gbps = (bytes_per_iter * 100) / (ms/1000.0) / 1e9;
    printf("%s : %g ms / 100 iters  (~%.2f GB/s)\n",
           cubin_path, ms, gbps);

    // Cleanup
    cuEventDestroy(start); cuEventDestroy(stop);
    cuMemFree(dA); cuMemFree(dB); cuMemFree(dC);
    cuModuleUnload(mod);
    cuCtxDestroy(ctx);
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <kernel.cubin> [N]\n", argv[0]);
        return 1;
    }
    const char* cubin_path = argv[1];
    int N = (argc >= 3) ? std::atoi(argv[2]) : (1<<26); // ~67M elems
    run_once(cubin_path, N);
    return 0;
}
