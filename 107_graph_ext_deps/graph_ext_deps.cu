#include <cuda_runtime.h>
#include <cstdio>

__global__ void kernel_a(float* data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    data[i] = data[i] + 1.0f;
}

__global__ void kernel_b(float* data){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    data[i] = data[i] * 2.0f;
}


int main(){
    int n = 1 << 20;
    float* d = nullptr;
    cudaMalloc(&d, n * sizeof(float));
    cudaMemset(d, 0, n * sizeof(float));

    cudaStream_t s1, s2;
    cudaStreamCreate(&s1);
    cudaStreamCreate(&s2);
    cudaEvent_t ev;
    cudaEventCreate(&ev);

    cudaGraph_t g1, g2;
    cudaGraphExec_t g1_exec, g2_exec;

    cudaStreamBeginCapture(s1, cudaStreamCaptureModeGlobal);
    kernel_a<<<(n+255)/256, 256, 0, s1>>>(d);
    cudaStreamEndCapture(s1, &g1);
    cudaGraphInstantiate(&g1_exec, g1, nullptr, nullptr, 0);

    cudaStreamBeginCapture(s2, cudaStreamCaptureModeGlobal);
    kernel_b<<<(n+255)/256, 256, 0, s2>>>(d);
    cudaStreamEndCapture(s2, &g2);
    cudaGraphInstantiate(&g2_exec, g2, nullptr, nullptr, 0);

    cudaGraphLaunch(g1_exec, s1);
    cudaEventRecord(ev, s1);
    cudaStreamWaitEvent(s2, ev, 0);
    cudaGraphLaunch(g2_exec, s2);

    cudaStreamSynchronize(s1);
    cudaStreamSynchronize(s2);
    printf("graphs done\n");

    cudaGraphExecDestroy(g1_exec);
    cudaGraphExecDestroy(g2_exec);
    cudaGraphDestroy(g1);
    cudaGraphDestroy(g2);
    cudaEventDestroy(ev);
    cudaStreamDestroy(s1);
    cudaStreamDestroy(s2);
    cudaFree(d);
    return 0;
}
