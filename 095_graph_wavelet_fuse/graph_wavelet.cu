#include <cuda_runtime.h>
#include <cstdio>

__global__ void pde_update(float* u, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) u[i] = u[i] * 0.99f;
}

__global__ void haar_step(const float* in, float* out, int n){
    int i = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (i + 1 >= n) return;
    float a = in[i];
    float b = in[i+1];
    out[i/2] = 0.5f * (a + b);
    out[n/2 + i/2] = 0.5f * (a - b);
}

int main(){
    int n = 1024;
    float* d_u=nullptr; float* d_tmp=nullptr;
    cudaMalloc(&d_u, n*sizeof(float));
    cudaMalloc(&d_tmp, n*sizeof(float));

    cudaGraph_t graph;
    cudaGraphExec_t graph_exec;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal);
    pde_update<<<(n+255)/256, 256, 0, stream>>>(d_u, n);
    haar_step<<<(n/2+255)/256, 256, 0, stream>>>(d_u, d_tmp, n);
    cudaStreamEndCapture(stream, &graph);

    cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);
    for (int i=0;i<100;i++) cudaGraphLaunch(graph_exec, stream);
    cudaStreamSynchronize(stream);

    printf("graph done\n");
    cudaGraphExecDestroy(graph_exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);
    cudaFree(d_u);
    cudaFree(d_tmp);
    return 0;
}
