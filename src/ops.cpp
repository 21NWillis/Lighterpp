#include "ops.h"
#include <math.h>

// output vector, input vector, weight matrix, number of rows/size of output vector, number of columns/size of input vector (same as width of weight matrix)
void naive_matmul(float* out, float* x, float *w, int n, int d) {
    for (int i = 0; i < n; i++) {
        float val = 0.0f;
        for (int k = 0; k < d; k++) {
            val += x[k] * w[i*d + k];
        }
        out[i] = val;
    }
    return;
}

// output vector, input vector, weight vector, size of vector
void RMSNorm(float* out, float* x, float* weight, int n) {
    float ss = 0.0f;
    for (int i = 0; i < n; i++) {
        ss += x[i] * x[i];
    }
    ss /= n;
    ss += 1e-5f; // Divide by zero protection
    ss = 1.0f / sqrtf(ss); 

    for (int i = 0; i < n; i++) {
        out[i] = x[i] * ss * weight[i];
    }

    return;
}
