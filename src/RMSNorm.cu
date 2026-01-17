#include "kernels.cuh"

#define BLOCK_SIZE 256

// Block size stride loop - accumulates x^2 for multiple elements
// Warp shuffle reduction - 1 partial sum per warp
// Store warp sums in shared memory
// First warp reduces warp sums from shared memory 
// Compute scale - rsqrtf(warp_total / n + 1e-5f)
// Block size stride loop - applies scale to output 
__global__ void rmsnorm(float* out, float* x, float* w, int n) {
    __shared__ float warp_sums[32];
    __shared__ float scale;

    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;
    int d4 = n / 4;

    float local_sum = 0.0f;
    for (int i = tid; i < d4; i += blockDim.x) {
        float4 x_vec = reinterpret_cast<float4*>(x)[i];
        local_sum += x_vec.x * x_vec.x;
        local_sum += x_vec.y * x_vec.y;
        local_sum += x_vec.z * x_vec.z;
        local_sum += x_vec.w * x_vec.w;
    }
    
    int base = d4 * 4;
    for (int i = base + tid; i < n; i += blockDim.x) {
        local_sum += x[i] * x[i];
    }

    
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 16);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 8);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 4);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 2);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 1);
    
    if (lane_id == 0) {
        warp_sums[warp_id] = local_sum;
    }

    __syncthreads();

    if (warp_id == 0) {
        
        float warp_total = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 16);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 8);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 4);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 2);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 1);
        
        if (lane_id == 0) {
            scale = rsqrtf(warp_total / n + 1e-5f);
        }
    }

    __syncthreads();
    
    for (int i = tid; i < n; i += blockDim.x) {
        out[i] = x[i] * scale * w[i];
    }
}

void cuda_rmsnorm(float* d_out, float* d_x, float* d_w, int n) {
    rmsnorm<<<1, BLOCK_SIZE>>>(d_out, d_x, d_w, n);
}
