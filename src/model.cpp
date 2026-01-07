#include "model.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "ops.h"

void malloc_run_state(RunState* s, Config* p) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;
    

    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;

    // ACTIVATION BUFFERS
    s->x = (float*)calloc(dim, sizeof(float));      // Input/State vector
    s->xb = (float*)calloc(dim, sizeof(float));     // Residual branch
    s->q = (float*)calloc(dim, sizeof(float));      // Query vector
    s->k = (float*)calloc(kv_dim, sizeof(float));   // Key vector
    s->v = (float*)calloc(kv_dim, sizeof(float));   // Value vector
    
    // FeedForward network buffers
    s->hb = (float*)calloc(hidden_dim, sizeof(float));
    s->he = (float*)calloc(hidden_dim, sizeof(float));

    //ATTENTION SCORES
    s->att = (float*)calloc(p->n_heads * p->seq_len, sizeof(float));

    //LOGITS
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));

    // KV CACHE
    int kv_cache_size = p->n_layers * p->seq_len * kv_dim;
    
    s->key_cache = (float*)calloc(kv_cache_size, sizeof(float));
    s->value_cache = (float*)calloc(kv_cache_size, sizeof(float));

    if (!s->x || !s->key_cache) {
        printf("Malloc failed! System out of memory?\n");
        exit(1);
    }
}

void free_run_state(RunState* s) {
    free(s->x);
    free(s->xb);
    free(s->hb);
    free(s->he);
    free(s->q);
    free(s->k);
    free(s->v);
    free(s->att);
    free(s->logits);
    free(s->key_cache);
    free(s->value_cache);
}


void attention(float* x, RunState* s, transformerWeights* w, Config* p, int layer, int pos) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = dim / p->n_heads;

    // 1. QKV Projections
    naive_matmul(s->q, x, w->wq, dim, dim);
    naive_matmul(s->k, x, w->wk, kv_dim, dim);
    naive_matmul(s->v, x, w->wv, kv_dim, dim);

    // 2. RoPE
    rope(s->q, s->k, pos, dim, kv_dim, head_size);
    
    // 3. KV Cache Update (Strided)
    int cache_stride = p->seq_len * head_size; // Size of one head's timeline
    for (int h = 0; h < p->n_kv_heads; h++) {
        int src_offset = h * head_size;
        int dst_offset = layer * (kv_dim * p->seq_len) + h * cache_stride + pos * head_size;
        
        memcpy(s->key_cache + dst_offset, s->k + src_offset, head_size * sizeof(float));
        memcpy(s->value_cache + dst_offset, s->v + src_offset, head_size * sizeof(float));
    }

    // 4. Multi-Head Attention Loop
    int gqa_factor = p->n_heads / p->n_kv_heads; // Grouped Query Attention factor

    for (int h = 0; h < p->n_heads; h++) {
        float* q_head = s->q + h * head_size;
        int kv_head = h / gqa_factor;

        int head_offset = layer * (kv_dim * p->seq_len) + kv_head * cache_stride;
        float* k_cache_head = s->key_cache + head_offset;
        float* v_cache_head = s->value_cache + head_offset;

        // Calculate Scores
        float* att_head = s->att + h * p->seq_len;
        naive_matmul(att_head, q_head, k_cache_head, pos + 1, head_size);

        // Scale scores
        float scale = 1.0f / sqrtf(head_size);
        for (int t = 0; t <= pos; t++) {
            att_head[t] *= scale;
        }

        // Softmax
        softmax(att_head, pos + 1);

        // Aggregation: V * Scores
        float* xb_head = s->xb + h * head_size;
        memset(xb_head, 0, head_size * sizeof(float));

        // Weighted sum
        for (int t = 0; t <= pos; t++) {
            float score = att_head[t];
            float* v_vec = v_cache_head + t * head_size;
            for (int i = 0; i < head_size; i++) {
                xb_head[i] += v_vec[i] * score;
            }
        }
    }

    // 5. Output Projection
    naive_matmul(x, s->xb, w->wo, dim, dim);
}