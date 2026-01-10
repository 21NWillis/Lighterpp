#include "ops.h"
#include <math.h>

#ifdef USE_CUDA
#include "kernels.cuh"
#endif

// output vector, input vector, weight matrix, number of rows/size of output vector, number of columns/size of input vector (same as width of weight matrix)
void naive_matmul(float* out, float* x, float *w, int n, int d) {
#ifdef USE_CUDA
    cuda_gemv(out, x, w, n, d);
#else
    for (int i = 0; i < n; i++) {
        float val = 0.0f;
        for (int k = 0; k < d; k++) {
            val += x[k] * w[i*d + k];
        }
        out[i] = val;
    }
#endif
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

void rope(float* q, float* k, int pos, int dim, int kv_dim, int head_size) {
    for (int i = 0; i < dim; i+=2) {
        int head_dim = i % head_size;
        float freq = 1.0f / powf(10000.0f, head_dim / (float)head_size);
        float val = pos * freq;
        float fcos = cosf(val);
        float fsin = sinf(val);

        float q0 = q[i];
        float q1 = q[i+1];
        q[i]   = q0 * fcos - q1 * fsin;
        q[i+1] = q0 * fsin + q1 * fcos;

        if (i < kv_dim) {
            float k0 = k[i];
            float k1 = k[i+1];
            k[i]   = k0 * fcos - k1 * fsin;
            k[i+1] = k0 * fsin + k1 * fcos;
        }
    }
}

void softmax(float* x, int size) {
    float max_val = x[0];
    for (int i = 1; i < size; i++) {
        if (x[i] > max_val) {
            max_val = x[i];
        }
    }

    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }

    for (int i = 0; i < size; i++) {
        x[i] = x[i] / sum;
    }

    return;
}

// SwiGLU Function
void swiglu(float* hb, float* h1, float* h3, int size) {
    for (int i = 0; i < size; i++) {
        float val = h1[i];

        float sig = 1.0f / (1.0f + expf(-val));
        
        hb[i] = (val * sig) * h3[i]; 
    }
    return;
}