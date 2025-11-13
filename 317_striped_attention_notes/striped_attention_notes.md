
# Striped Attention Notes

- Ring Attention partitions contiguous blocks; Striped Attention interleaves tokens across devices.
- Striped layout improves load balance when attention is non-uniform across sequence.
- Trade-off: more complex indexing and potential higher communication cost.
- Contiguous blocks are cache-friendly; striped blocks distribute memory pressure.

Implementation idea:
- Split tokens into even/odd indices (or k-way stripes).
- Each stripe computes attention within itself, then combine results or refine with cross-stripe passes.
