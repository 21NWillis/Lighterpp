#include "kernels.cuh"

#define BLOCK_SIZE 256


__global__ void swiglu_kernel(float* hb, float* h1, float* h3, int size) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i >= size / 4) return;

    float4 h1_val = reinterpret_cast<float4*>(h1)[i];
    float4 h3_val = reinterpret_cast<float4*>(h3)[i];

    float4 out;

    out.x = h1_val.x / (1 + __expf(-h1_val.x));
    out.y = h1_val.y / (1 + __expf(-h1_val.y));
    out.z = h1_val.z / (1 + __expf(-h1_val.z));
    out.w = h1_val.w / (1 + __expf(-h1_val.w));

    out.x *= h3_val.x;
    out.y *= h3_val.y;
    out.z *= h3_val.z;
    out.w *= h3_val.w;

    reinterpret_cast<float4*>(hb)[i] = out;
}

// Host wrapper function
void cuda_swiglu(float* d_hb, float* d_h1, float* d_h3, int size) {
    int num_float4s = size / 4;
    int num_blocks = (num_float4s + BLOCK_SIZE - 1) / BLOCK_SIZE;
    swiglu_kernel<<<num_blocks, BLOCK_SIZE>>>(d_hb, d_h1, d_h3, size);
}
