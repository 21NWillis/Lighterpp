#include "kernels.cuh"
#include <cuda_runtime.h>

#define BLOCK_SIZE 256

__global__ void gemv_kernel(float* out, float* x, float* w, int n, int d) {
    __shared__ float warp_sums[32];
    // Use defined limit (40KB=10240 floats)
    __shared__ float x_shared[MAX_SHARED_FLOATS];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    int lane_id = tid % 32;
    int warp_id = tid / 32;
    int num_warps = blockDim.x / 32;

    for (int i = tid; i < d; i += blockDim.x) {
        x_shared[i] = x[i];
    }

    __syncthreads();
    
    float* w_row = w + row * d;
    
    float local_sum = 0.0f;
    int d4 = d / 4;
    
    for (int i = tid; i < d4; i += blockDim.x) {
        float4 w_vec = reinterpret_cast<float4*>(w_row)[i];
        float4 x_vec = reinterpret_cast<float4*>(x_shared)[i];
        
        local_sum += w_vec.x * x_vec.x;
        local_sum += w_vec.y * x_vec.y;
        local_sum += w_vec.z * x_vec.z;
        local_sum += w_vec.w * x_vec.w;
    }
    int base = d4 * 4;
    for (int i = base + tid; i < d; i += blockDim.x) {
        local_sum += w_row[i] * x_shared[i];
    }

    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 16);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 8);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 4);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 2);
    local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, 1);
    
    if (lane_id == 0) warp_sums[warp_id] = local_sum;
    
    __syncthreads();
    
    if (warp_id == 0) {
        float warp_total = (lane_id < num_warps) ? warp_sums[lane_id] : 0.0f;
        
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 16);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 8);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 4);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 2);
        warp_total += __shfl_down_sync(0xFFFFFFFF, warp_total, 1);
        
        if (lane_id == 0) {
            out[row] = warp_total;
        }
    }
}

// Host wrapper function
void cuda_gemv(float* d_out, float* d_x, float* d_w, int n, int d) {
    gemv_kernel<<<n, BLOCK_SIZE>>>(d_out, d_x, d_w, n, d);
}
