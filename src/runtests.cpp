#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "ops.h"
#include "tensor.h"

#define GREEN "\033[0;32m"
#define RED   "\033[0;31m"
#define RESET "\033[0m"

int is_close(float a, float b) {
    return fabsf(a - b) < 1e-4f;
}

void test_matmul_square() {
    printf("Test: Square MatMul (Vector-Matrix)... ");
    
    float x[4] = {1, 2, 3, 4}; 
    float w[4] = {1, 0, 0, 1};
    float out[4] = {0};

    naive_matmul(out, x, w, 2, 2); 

    if (is_close(out[0], 1.0f) && is_close(out[1], 2.0f)) {
        printf(GREEN "PASSED" RESET "\n");
    } else {
        printf(RED "FAILED" RESET " (Got %f, %f, Expected 1.0, 2.0)\n", out[0], out[1]);
    }
}

void test_matmul_rectangular() {
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
    } else {
        printf(RED "FAILED" RESET " (Got %f, %f, Expected 4.5, 5.0)\n", out[0], out[1]);
    }
}

int main() {
    printf("--- Lighter++ Unit Tests ---\n");
    test_matmul_square();
    test_matmul_rectangular();
    return 0;
}