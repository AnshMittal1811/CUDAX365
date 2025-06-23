#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

__global__ void add_kernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] += 1.0f;
    }
}

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    int iters = (argc > 2) ? std::atoi(argv[2]) : 100;

    float *d_data = nullptr;
    cudaMalloc(&d_data, n * sizeof(float));

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    add_kernel<<<blocks, threads, 0, stream>>>(d_data, n);
    cudaStreamEndCapture(stream, &graph);
    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

    for (int i = 0; i < iters; ++i) {
        cudaGraphLaunch(graph_exec, stream);
    }
    cudaStreamSynchronize(stream);

    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_data);

    std::printf("graph_worker done\n");
    return 0;
}
