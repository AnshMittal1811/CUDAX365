#include <cuda_runtime.h>
#include <cstdio>

texture<float, cudaTextureType3D, cudaReadModeElementType> tex3d;

__global__ void sample_kernel(float* out, int n){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float u = (i % 128) / 127.0f;
    float v = ((i / 128) % 128) / 127.0f;
    float w = (i / (128 * 128)) / 127.0f;
    out[i] = tex3D(tex3d, u, v, w);
}

int main(){
    int n = 128 * 128 * 128;
    cudaArray* arr;
    cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    cudaExtent extent = make_cudaExtent(128, 128, 128);
    cudaMalloc3DArray(&arr, &desc, extent);

    cudaMemcpy3DParms params = {0};
    params.dstArray = arr;
    params.extent = extent;
    params.kind = cudaMemcpyHostToDevice;
    float* host = new float[n];
    for (int i=0;i<n;i++) host[i] = (i % 255) / 255.0f;
    cudaMemcpy3DParms copy = {0};
    copy.srcPtr = make_cudaPitchedPtr(host, 128*sizeof(float), 128, 128);
    copy.dstArray = arr;
    copy.extent = extent;
    copy.kind = cudaMemcpyHostToDevice;
    cudaMemcpy3D(&copy);

    cudaBindTextureToArray(tex3d, arr, desc);

    float* d_out;
    cudaMalloc(&d_out, n * sizeof(float));
    sample_kernel<<<(n+255)/256, 256>>>(d_out, n);
    cudaDeviceSynchronize();

    cudaFree(d_out);
    cudaUnbindTexture(tex3d);
    cudaFreeArray(arr);
    delete[] host;
    printf("sampling done\n");
    return 0;
}
