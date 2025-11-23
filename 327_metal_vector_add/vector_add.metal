
#include <metal_stdlib>
using namespace metal;

kernel void vector_add(
    const device float* a [[buffer(0)]],
    const device float* b [[buffer(1)]],
    device float* out [[buffer(2)]],
    uint id [[thread_position_in_grid]]) {
    out[id] = a[id] + b[id];
}
