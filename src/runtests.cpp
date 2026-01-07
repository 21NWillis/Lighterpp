#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "ops.h"
#include "tensor.h"
#include "model.h"

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
    attention(x, &s, &w, &config, 0, 0);
    
    // Run Attention at pos 1
    attention(x, &s, &w, &config, 0, 1);
    
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

int main() {
    printf("--- Lighter++ Unit Tests ---\n");
    int failures = 0;
    failures += test_matmul_square();
    failures += test_matmul_rectangular();
    failures += test_RMSNorm();
    failures += test_rope();
    failures += test_softmax();
    failures += test_attention_smoke();
    
    if (failures == 0) {
        printf("ALL TESTS PASSED\n");
    } else {
        printf("%d TESTS FAILED\n", failures);
    }
    return failures;
}