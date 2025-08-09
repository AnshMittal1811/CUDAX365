#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

int main(int argc, char **argv) {
    int n = (argc > 1) ? std::atoi(argv[1]) : (1 << 20);
    size_t bytes = n * sizeof(float);

    float *h = nullptr;
    cudaHostAlloc(&h, bytes, cudaHostAllocDefault);
    float *d = nullptr;
    cudaMalloc(&d, bytes);

    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaGraphCreate(&graph, 0);

    cudaGraphNode_t memcpy_node;
    cudaMemcpy3DParms params = {0};
    params.srcPtr = make_cudaPitchedPtr(h, bytes, n, 1);
    params.dstPtr = make_cudaPitchedPtr(d, bytes, n, 1);
    params.extent = make_cudaExtent(bytes, 1, 1);
    params.kind = cudaMemcpyHostToDevice;

    cudaGraphAddMemcpyNode(&memcpy_node, graph, nullptr, 0, &params);

    cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
    cudaGraphLaunch(exec, stream);
    cudaStreamSynchronize(stream);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    cudaFree(d);
    cudaFreeHost(h);

    std::printf("graph memcpy done\n");
    return 0;
}
