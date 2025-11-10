
# Ring Attention Notes

- Goal: scale attention to very long contexts by distributing work across GPUs in a ring.
- Key idea: each device holds a block of tokens and streams K/V blocks around the ring.
- Compute: each block of queries attends to local K/V plus streamed blocks, accumulating partial softmax.
- Communication: ring-based all-reduce style to share K/V blocks without all-to-all.
- Benefit: memory per GPU is reduced; compute overlaps with communication.

Takeaways for this project:
- Blockwise attention on one GPU can emulate the ring pattern with sequential blocks.
- Accuracy depends on how much context is retained per block and how softmax is accumulated.
