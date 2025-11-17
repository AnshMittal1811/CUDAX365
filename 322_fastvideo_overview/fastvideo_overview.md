
# FastVideo Overview Notes

- Modular video diffusion pipeline with optimized attention and scheduling.
- Uses tiled or sliding-window attention to reduce memory and compute.
- Focuses on throughput (frames/sec) and memory efficiency for long clips.
- Emphasizes kernel fusion and mixed precision to keep GPU utilization high.

Next steps:
- Run a minimal example to measure frames/sec.
- Inspect attention kernels for memory/compute tiling patterns.
