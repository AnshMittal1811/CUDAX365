RWStructuredBuffer<float> Out : register(u0);
StructuredBuffer<float> In : register(t0);
cbuffer Params : register(b0) { int width; int height; };

[numthreads(16, 16, 1)]
void main(uint3 tid : SV_DispatchThreadID){
    if (tid.x >= width || tid.y >= height) return;
    uint idx = tid.y * width + tid.x;
    float u0 = In[idx];
    Out[idx] = u0 * 0.99;
}

