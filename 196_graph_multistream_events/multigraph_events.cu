#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void kernel_a(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

__global__ void kernel_b(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= 1.0001f;
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);

    float *d_data = nullptr;
    cudaMalloc(&d_data, n * sizeof(float));

    cudaStream_t stream1, stream2;
    cudaStreamCreate(&stream1);
    cudaStreamCreate(&stream2);

    cudaEvent_t event;
    cudaEventCreate(&event);

    cudaGraph_t graph1, graph2;
    cudaGraphExec_t exec1, exec2;

    cudaStreamBeginCapture(stream1, cudaStreamCaptureModeGlobal);
    kernel_a<<<(n + 255) / 256, 256, 0, stream1>>>(d_data, n);
    cudaEventRecord(event, stream1);
    cudaStreamEndCapture(stream1, &graph1);

    cudaStreamBeginCapture(stream2, cudaStreamCaptureModeGlobal);
    cudaStreamWaitEvent(stream2, event, 0);
    kernel_b<<<(n + 255) / 256, 256, 0, stream2>>>(d_data, n);
    cudaStreamEndCapture(stream2, &graph2);

    cudaGraphInstantiate(&exec1, graph1, nullptr, nullptr, 0);
    cudaGraphInstantiate(&exec2, graph2, nullptr, nullptr, 0);

    for (int i = 0; i < 10; ++i) {
        cudaGraphLaunch(exec1, stream1);
        cudaGraphLaunch(exec2, stream2);
    }

    cudaStreamSynchronize(stream1);
    cudaStreamSynchronize(stream2);

    cudaGraphExecDestroy(exec1);
    cudaGraphExecDestroy(exec2);
    cudaGraphDestroy(graph1);
    cudaGraphDestroy(graph2);
    cudaEventDestroy(event);
    cudaStreamDestroy(stream1);
    cudaStreamDestroy(stream2);
    cudaFree(d_data);

    std::printf("multigraph events done\n");
    return 0;
}
