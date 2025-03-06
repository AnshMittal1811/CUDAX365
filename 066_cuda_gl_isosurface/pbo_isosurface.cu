#include <cstdio>
#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <GL/glew.h>
#include <GLFW/glfw3.h>

__global__ void iso_pbo(uchar4* pbo, int w, int h){
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int idx = y * w + x;
    int v = ((x-320)*(x-320) + (y-240)*(y-240)) < 10000 ? 255 : 0;
    pbo[idx] = make_uchar4(v, v, v, 255);
}

int main(){
    if (!glfwInit()) return 1;
    GLFWwindow* window = glfwCreateWindow(640, 480, "Iso", nullptr, nullptr);
    glfwMakeContextCurrent(window);
    glewInit();

    GLuint pbo = 0;
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 640 * 480 * sizeof(uchar4), nullptr, GL_DYNAMIC_DRAW);

    cudaGraphicsResource* cuda_pbo = nullptr;
    cudaGraphicsGLRegisterBuffer(&cuda_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);

    cudaGraphicsMapResources(1, &cuda_pbo);
    uchar4* dev_ptr = nullptr;
    size_t size = 0;
    cudaGraphicsResourceGetMappedPointer((void**)&dev_ptr, &size, cuda_pbo);

    dim3 block(16,16), grid((640+15)/16, (480+15)/16);
    iso_pbo<<<grid, block>>>(dev_ptr, 640, 480);
    cudaDeviceSynchronize();

    cudaGraphicsUnmapResources(1, &cuda_pbo);

    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glDrawPixels(640, 480, GL_RGBA, GL_UNSIGNED_BYTE, 0);
    glfwSwapBuffers(window);
    glfwPollEvents();

    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}
