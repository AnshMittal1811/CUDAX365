#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <chrono>

__global__ void noop_kernel() {}

static float run_sync(unsigned int flags, int iters) {
    cudaEvent_t event;
    cudaEventCreateWithFlags(&event, flags);
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        noop_kernel<<<1, 1>>>();
        cudaEventRecord(event, 0);
        cudaEventSynchronize(event);
    }
    auto end = std::chrono::high_resolution_clock::now();
    cudaEventDestroy(event);
    auto us = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
    return static_cast<float>(us) / iters;
}

int main(int argc, char **argv) {
    int iters = (argc > 1) ? std::atoi(argv[1]) : 1000;
    float default_us = run_sync(cudaEventDefault, iters);
    float blocking_us = run_sync(cudaEventBlockingSync, iters);

    std::printf("default_event_us=%.3f\n", default_us);
    std::printf("blocking_event_us=%.3f\n", blocking_us);
    return 0;
}
