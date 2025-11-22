
# Metal Compute Notes

- Metal kernels run on Apple GPUs using a threadgroup model similar to CUDA blocks.
- `thread_position_in_grid` maps to global thread index.
- Buffers are `device` pointers; threadgroup memory is local/shared.
- Command buffers enqueue compute pipelines and synchronize with completion handlers.
