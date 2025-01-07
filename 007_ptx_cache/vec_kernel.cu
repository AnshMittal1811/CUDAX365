// vec_kernel.cu  (device code only)
extern "C" __global__
void vec_add_kernel(const float* __restrict__ A,
                    const float* __restrict__ B,
                    float* __restrict__ C,
                    int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // Intentionally simple so PTX shows ld.global/st.global
        C[i] = A[i] + B[i];
    }
}