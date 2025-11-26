
# AMD Composable Kernel Notes

- CK uses template meta-programming to generate specialized kernels.
- Kernels are assembled from tensor operations and layout primitives.
- MIOpen integrates CK for high-performance conv/GEMM variants.
