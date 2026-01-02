#include "ops.h"

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