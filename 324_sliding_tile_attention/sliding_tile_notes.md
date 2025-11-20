
# Sliding Tile Attention Notes

- Split the attention matrix into tiles that slide over the sequence.
- Each tile computes local attention to reduce O(N^2) memory cost.
- Overlapping tiles preserve context while bounding compute per token.
- Typically combined with streaming or ring communication.
