#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "ops.h"
#include "tensor.h"
#include "model.h"

#ifdef USE_CUDA
#include "kernels.cuh"
#include <cuda_runtime.h>
#endif

#define GREEN "\033[0;32m"
#define RED   "\033[0;31m"
#define RESET "\033[0m"

int test_attention_smoke() {
    printf("Test: Attention Smoke Test (No crash)... ");
    
    // Setup Config
    Config config;
    config.dim = 32;
    config.hidden_dim = 64;
    config.n_layers = 1;
    config.n_heads = 4;
    config.n_kv_heads = 4;
    config.vocab_size = 100;
    config.seq_len = 10;
    
    // Setup Weights
    transformerWeights w;
    int dim = config.dim;
    w.wq = (float*)calloc(dim * dim, sizeof(float));
    w.wk = (float*)calloc(dim * dim, sizeof(float));
    w.wv = (float*)calloc(dim * dim, sizeof(float));
    w.wo = (float*)calloc(dim * dim, sizeof(float));
    
    // Setup State
    RunState s;
    malloc_run_state(&s, &config);
    
    // Setup Input
    float* x = (float*)calloc(dim, sizeof(float));
    for(int i=0; i<dim; i++) x[i] = 0.5f;
    
    // Run Attention at pos 0
    attention(s.xb2, s.xb, &s, &w, &config, 0, 0);
    
    // Run Attention at pos 1
    attention(s.xb2, s.xb, &s, &w, &config, 0, 1);
    
    // Cleanup
    free(w.wq); free(w.wk); free(w.wv); free(w.wo);
    free_run_state(&s);
    free(x);
    
    printf(GREEN "PASSED" RESET "\n");
    return 0;
}

int is_close(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

// Relaxed tolerance for CUDA tests (GPU float ops can have slight differences)
int is_close_gpu(float a, float b) {
    return fabsf(a - b) < 1e-3f;  // 0.1% tolerance
}

int test_matmul_square() {
    printf("Test: Square MatMul (Vector-Matrix)... ");
    
    float x[4] = {1, 2, 3, 4}; 
    float w[4] = {1, 0, 0, 1};
    float out[4] = {0};

    naive_matmul(out, x, w, 2, 2); 

    if (is_close(out[0], 1.0f) && is_close(out[1], 2.0f)) {
        printf(GREEN "PASSED" RESET "\n");
        return 0;
    } else {
        printf(RED "FAILED" RESET " (Got %f, %f, Expected 1.0, 2.0)\n", out[0], out[1]);
        return 1;
    }
}

int test_matmul_rectangular() {
    printf("Test: Rectangular MatMul... ");
    
    float x[3] = {1, 2, 3};
    
    float w[6] = {
        0.5, 0.5, 1.0, 
        2.0, 0.0, 1.0
    };
    
    float out[2] = {0};
    
    naive_matmul(out, x, w, 2, 3);
    
    if (is_close(out[0], 4.5f) && is_close(out[1], 5.0f)) {
        printf(GREEN "PASSED" RESET "\n");
        return 0;
    } else {
        printf(RED "FAILED" RESET " (Got %f, %f, Expected 4.5, 5.0)\n", out[0], out[1]);
        return 1;
    }
}

int test_RMSNorm() {
    printf("Test: RMSNorm... ");
    float x[4] = {2, 2, 2, 2};
    float weight[4] = {1, 1, 1, 1};
    
    float out[4] = {0};

    RMSNorm(out, x, weight, 4);

    if (is_close(out[0], 1.0f) && is_close(out[1], 1.0f) && is_close(out[2], 1.0f) && is_close(out[3], 1.0f)) {
        printf(GREEN "PASSED" RESET "\n");
        return 0;
    } else {
        printf(RED "FAILED" RESET "\n");
        return 1;
    }
}


int test_rope() {
    int failures = 0;
    printf("Test: RoPE (Position 0 Identity)... ");
    int dim = 4;
    int kv_dim = 4;
    int head_size = 2;
    
    float q[4] = {1, 2, 3, 4};
    float k[4] = {1, 2, 3, 4};
    
    rope(q, k, 0, dim, kv_dim, head_size);
    
    bool passed = true;
    for(int i=0; i<4; i++) {
        if (!is_close(q[i], (float)(i+1))) passed = false;
    }
    
    if (passed) {
        printf(GREEN "PASSED" RESET "\n");
    } else {
        printf(RED "FAILED" RESET "\n");
        failures++;
    }

    printf("Test: RoPE (Position 1 Movement)... ");
    float q2[4] = {1, 1, 1, 1};
    float k2[4] = {1, 1, 1, 1};
    
    rope(q2, k2, 1, 4, 4, 2);
    
    if (!is_close(q2[0], 1.0f) && !is_close(q2[1], 1.0f)) {
        printf(GREEN "PASSED" RESET "\n");
    } else {
        printf(RED "FAILED" RESET " (Vectors did not rotate)\n");
        failures++;
    }
    return failures;
}

int test_transformer_block_integration() {
    printf("Test: Transformer Block Integration... ");
    
    // Setup Config
    Config config;
    config.dim = 64;
    config.hidden_dim = 128;
    config.n_layers = 2;
    config.n_heads = 4;
    config.n_kv_heads = 4;
    config.vocab_size = 100;
    config.seq_len = 10;
    
    // Setup Weights (all zeros for simplicity)
    transformerWeights w;
    int dim = config.dim;
    int hidden_dim = config.hidden_dim;
    
    w.rms_att_weight = (float*)calloc(dim, sizeof(float));
    w.wq = (float*)calloc(dim * dim, sizeof(float));
    w.wk = (float*)calloc(dim * dim, sizeof(float));
    w.wv = (float*)calloc(dim * dim, sizeof(float));
    w.wo = (float*)calloc(dim * dim, sizeof(float));
    w.rms_ffn_weight = (float*)calloc(dim, sizeof(float));
    w.w1 = (float*)calloc(hidden_dim * dim, sizeof(float));
    w.w2 = (float*)calloc(dim * hidden_dim, sizeof(float));
    w.w3 = (float*)calloc(hidden_dim * dim, sizeof(float));
    
    // Initialize norm weights to 1.0
    for (int i = 0; i < dim; i++) {
        w.rms_att_weight[i] = 1.0f;
        w.rms_ffn_weight[i] = 1.0f;
    }
    
    // Setup State
    RunState s;
    malloc_run_state(&s, &config);
    
    // Setup Input (simple pattern)
    float* x = (float*)calloc(dim, sizeof(float));
    for(int i = 0; i < dim; i++) x[i] = 0.1f;
    
    // Run transformer block (should not crash)
    transformer_block(x, &s, &w, &config, 0, 0);
    
    // Verify output is finite (not NaN or Inf)
    bool all_finite = true;
    for (int i = 0; i < dim; i++) {
        if (!isfinite(x[i])) {
            all_finite = false;
            break;
        }
    }
    
    // Cleanup
    free(w.rms_att_weight);
    free(w.wq); free(w.wk); free(w.wv); free(w.wo);
    free(w.rms_ffn_weight);
    free(w.w1); free(w.w2); free(w.w3);
    free_run_state(&s);
    free(x);
    
    if (all_finite) {
        printf(GREEN "PASSED" RESET "\n");
        return 0;
    } else {
        printf(RED "FAILED" RESET " (NaN or Inf detected)\n");
        return 1;
    }
}

int test_softmax() {
    printf("Test: Softmax... ");
    float x[3] = {1.0f, 2.0f, 3.0f};
    // exp(1-3) = e^-2 = 0.1353
    // exp(2-3) = e^-1 = 0.3678
    // exp(3-3) = e^0  = 1.0000
    // sum = 1.5031
    // probs: 0.090, 0.244, 0.665

    softmax(x, 3);

    float sum = x[0] + x[1] + x[2];
    
    // Check if sum is ~1.0
    if (!is_close(sum, 1.0f)) {
        printf(RED "FAILED" RESET " (Sum = %f)\n", sum);
        return 1;
    }

    // Check individual values
    if (is_close(x[0], 0.09003f) && is_close(x[1], 0.24473f) && is_close(x[2], 0.66524f)) {
        printf(GREEN "PASSED" RESET "\n");
        return 0;
    } else {
        printf(RED "FAILED" RESET " (Got %f, %f, %f)\n", x[0], x[1], x[2]);
        return 1;
    }
}

int test_swiglu() {
    printf("Test: SwiGLU... ");
    float hb[2];
    float h1[2] = {1.0f, -1.0f}; // Gates
    float h3[2] = {2.0f, 3.0f};  // Values
    
    // SiLU(1.0) = 1.0 * sigmoid(1.0) = 1.0 * 0.73105 = 0.73105
    // SwiGLU[0] = 0.73105 * 2.0 = 1.46211
    
    // SiLU(-1.0) = -1.0 * sigmoid(-1.0) = -1.0 * 0.26894 = -0.26894
    // SwiGLU[1] = -0.26894 * 3.0 = -0.80682

    swiglu(hb, h1, h3, 2);

    if (is_close(hb[0], 1.46211f) && is_close(hb[1], -0.80682f)) {
         printf(GREEN "PASSED" RESET "\n");
         return 0;
    } else {
         printf(RED "FAILED" RESET " (Got %f, %f)\n", hb[0], hb[1]);
         return 1;
    }
}

// =============================================================================
// CUDA Tests - Compare CUDA kernel outputs against CPU reference
// =============================================================================
#ifdef USE_CUDA

int test_cuda_gemv() {
    printf("Test: CUDA GEMV... ");
    
    const int n = 256;  // output dim
    const int d = 512;  // input dim (typical model sizes)
    
    float* x = (float*)malloc(d * sizeof(float));
    float* w = (float*)malloc(n * d * sizeof(float));
    float* out_cpu = (float*)malloc(n * sizeof(float));
    float* out_gpu = (float*)malloc(n * sizeof(float));
    
    // Initialize with simple pattern
    for (int i = 0; i < d; i++) x[i] = 0.1f * (i % 10);
    for (int i = 0; i < n * d; i++) w[i] = 0.01f * (i % 100);
    
    // CPU reference
    naive_matmul(out_cpu, x, w, n, d);
    
    // GPU version
    float *d_x, *d_w, *d_out;
    cudaMalloc(&d_x, d * sizeof(float));
    cudaMalloc(&d_w, n * d * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    
    cudaMemcpy(d_x, x, d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, n * d * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_out, 0, n * sizeof(float));
    
    cuda_gemv(d_out, d_x, d_w, n, d);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf(RED "FAILED" RESET " (CUDA error: %s)\n", cudaGetErrorString(err));
        cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
        free(x); free(w); free(out_cpu); free(out_gpu);
        return 1;
    }
    
    cudaMemcpy(out_gpu, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare
    bool passed = true;
    float max_diff = 0.0f;
    int fail_idx = -1;
    for (int i = 0; i < n; i++) {
        float diff = fabsf(out_cpu[i] - out_gpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
            fail_idx = i;
        }
        if (!is_close_gpu(out_cpu[i], out_gpu[i])) {
            passed = false;
        }
    }
    
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
    
    if (passed) {
        free(x); free(w); free(out_cpu); free(out_gpu);
        printf(GREEN "PASSED" RESET "\n");
        return 0;
    } else {
        printf(RED "FAILED" RESET " (max diff=%f at idx=%d: cpu=%f gpu=%f)\n", 
               max_diff, fail_idx, out_cpu[fail_idx], out_gpu[fail_idx]);
        free(x); free(w); free(out_cpu); free(out_gpu);
        return 1;
    }
}

int test_cuda_rmsnorm() {
    printf("Test: CUDA RMSNorm... ");
    
    const int n = 256;
    float* x = (float*)malloc(n * sizeof(float));
    float* w = (float*)malloc(n * sizeof(float));
    float* out_cpu = (float*)malloc(n * sizeof(float));
    float* out_gpu = (float*)malloc(n * sizeof(float));
    
    for (int i = 0; i < n; i++) {
        x[i] = 0.5f + 0.1f * (i % 10);
        w[i] = 1.0f;
    }
    
    // CPU reference
    RMSNorm(out_cpu, x, w, n);
    
    // GPU version
    float *d_x, *d_w, *d_out;
    cudaMalloc(&d_x, n * sizeof(float));
    cudaMalloc(&d_w, n * sizeof(float));
    cudaMalloc(&d_out, n * sizeof(float));
    
    cudaMemcpy(d_x, x, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_w, w, n * sizeof(float), cudaMemcpyHostToDevice);
    
    cuda_rmsnorm(d_out, d_x, d_w, n);
    cudaDeviceSynchronize();
    
    cudaMemcpy(out_gpu, d_out, n * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool passed = true;
    for (int i = 0; i < n; i++) {
        if (!is_close_gpu(out_cpu[i], out_gpu[i])) {
            passed = false;
            break;
        }
    }
    
    cudaFree(d_x); cudaFree(d_w); cudaFree(d_out);
    free(x); free(w); free(out_cpu); free(out_gpu);
    
    if (passed) {
        printf(GREEN "PASSED" RESET "\n");
        return 0;
    } else {
        printf(RED "FAILED" RESET " (Output mismatch)\n");
        return 1;
    }
}

int test_cuda_rope() {
    printf("Test: CUDA RoPE... ");
    
    const int dim = 128;
    const int kv_dim = 128;
    const int head_size = 64;
    const int pos = 5;
    
    float* q_cpu = (float*)malloc(dim * sizeof(float));
    float* k_cpu = (float*)malloc(kv_dim * sizeof(float));
    float* q_gpu = (float*)malloc(dim * sizeof(float));
    float* k_gpu = (float*)malloc(kv_dim * sizeof(float));
    
    for (int i = 0; i < dim; i++) q_cpu[i] = q_gpu[i] = 1.0f + 0.1f * i;
    for (int i = 0; i < kv_dim; i++) k_cpu[i] = k_gpu[i] = 0.5f + 0.1f * i;
    
    // CPU reference
    rope(q_cpu, k_cpu, pos, dim, kv_dim, head_size);
    
    // GPU version
    float *d_q, *d_k;
    cudaMalloc(&d_q, dim * sizeof(float));
    cudaMalloc(&d_k, kv_dim * sizeof(float));
    
    cudaMemcpy(d_q, q_gpu, dim * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_k, k_gpu, kv_dim * sizeof(float), cudaMemcpyHostToDevice);
    
    cuda_rope(d_q, d_k, pos, dim, kv_dim, head_size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(q_gpu, d_q, dim * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(k_gpu, d_k, kv_dim * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool passed = true;
    for (int i = 0; i < dim; i++) {
        if (!is_close_gpu(q_cpu[i], q_gpu[i])) {
            passed = false;
            break;
        }
    }
    for (int i = 0; i < kv_dim && passed; i++) {
        if (!is_close_gpu(k_cpu[i], k_gpu[i])) {
            passed = false;
            break;
        }
    }
    
    cudaFree(d_q); cudaFree(d_k);
    free(q_cpu); free(k_cpu); free(q_gpu); free(k_gpu);
    
    if (passed) {
        printf(GREEN "PASSED" RESET "\n");
        return 0;
    } else {
        printf(RED "FAILED" RESET " (Output mismatch)\n");
        return 1;
    }
}

int test_cuda_swiglu() {
    printf("Test: CUDA SwiGLU... ");
    
    const int size = 256;
    float* h1 = (float*)malloc(size * sizeof(float));
    float* h3 = (float*)malloc(size * sizeof(float));
    float* out_cpu = (float*)malloc(size * sizeof(float));
    float* out_gpu = (float*)malloc(size * sizeof(float));
    
    for (int i = 0; i < size; i++) {
        h1[i] = -2.0f + 0.02f * i;  // Range from -2 to +3
        h3[i] = 1.0f + 0.01f * i;
    }
    
    // CPU reference
    swiglu(out_cpu, h1, h3, size);
    
    // GPU version
    float *d_h1, *d_h3, *d_out;
    cudaMalloc(&d_h1, size * sizeof(float));
    cudaMalloc(&d_h3, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    
    cudaMemcpy(d_h1, h1, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_h3, h3, size * sizeof(float), cudaMemcpyHostToDevice);
    
    cuda_swiglu(d_out, d_h1, d_h3, size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(out_gpu, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    bool passed = true;
    for (int i = 0; i < size; i++) {
        if (!is_close_gpu(out_cpu[i], out_gpu[i])) {
            passed = false;
            break;
        }
    }
    
    cudaFree(d_h1); cudaFree(d_h3); cudaFree(d_out);
    free(h1); free(h3); free(out_cpu); free(out_gpu);
    
    if (passed) {
        printf(GREEN "PASSED" RESET "\n");
        return 0;
    } else {
        printf(RED "FAILED" RESET " (Output mismatch)\n");
        return 1;
    }
}

int test_cuda_softmax() {
    printf("Test: CUDA Softmax... ");
    
    const int size = 256;
    float* x_cpu = (float*)malloc(size * sizeof(float));
    float* x_gpu = (float*)malloc(size * sizeof(float));
    float* out_gpu = (float*)malloc(size * sizeof(float));
    
    // Initialize with range of values to test numerical stability
    for (int i = 0; i < size; i++) {
        x_cpu[i] = x_gpu[i] = -5.0f + 0.04f * i;  // Range from -5 to +5
    }
    
    // CPU reference (in-place)
    softmax(x_cpu, size);
    
    // GPU version
    float *d_x, *d_out;
    cudaMalloc(&d_x, size * sizeof(float));
    cudaMalloc(&d_out, size * sizeof(float));
    
    cudaMemcpy(d_x, x_gpu, size * sizeof(float), cudaMemcpyHostToDevice);
    
    cuda_softmax(d_out, d_x, size);
    cudaDeviceSynchronize();
    
    cudaMemcpy(out_gpu, d_out, size * sizeof(float), cudaMemcpyDeviceToHost);
    
    // Compare
    bool passed = true;
    float max_diff = 0.0f;
    int fail_idx = -1;
    for (int i = 0; i < size; i++) {
        float diff = fabsf(x_cpu[i] - out_gpu[i]);
        if (diff > max_diff) {
            max_diff = diff;
            fail_idx = i;
        }
        if (!is_close_gpu(x_cpu[i], out_gpu[i])) {
            passed = false;
        }
    }
    
    // Also verify sum is ~1.0
    float sum = 0.0f;
    for (int i = 0; i < size; i++) sum += out_gpu[i];
    if (fabsf(sum - 1.0f) > 1e-3f) {
        passed = false;
    }
    
    cudaFree(d_x); cudaFree(d_out);
    
    if (passed) {
        free(x_cpu); free(x_gpu); free(out_gpu);
        printf(GREEN "PASSED" RESET "\n");
        return 0;
    } else {
        printf(RED "FAILED" RESET " (max diff=%f at idx=%d, sum=%f)\n", max_diff, fail_idx, sum);
        free(x_cpu); free(x_gpu); free(out_gpu);
        return 1;
    }
}

#endif // USE_CUDA

int main() {
    printf("--- Lighter++ Unit Tests ---\n");
    int failures = 0;
    failures += test_matmul_square();
    failures += test_matmul_rectangular();
    failures += test_RMSNorm();
    failures += test_rope();
    failures += test_softmax();
    failures += test_swiglu();
    
#ifdef USE_CUDA
    printf("\n--- CUDA Kernel Tests ---\n");
    failures += test_cuda_gemv();
    failures += test_cuda_rmsnorm();
    failures += test_cuda_rope();
    failures += test_cuda_swiglu();
    failures += test_cuda_softmax();
    printf("\n--- Integration Tests ---\n");
#endif

    failures += test_attention_smoke();
    failures += test_transformer_block_integration();
    
    if (failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d TESTS FAILED\n", failures);
    }
    return failures;
}