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
    s->xb2 = (float*)calloc(dim, sizeof(float));    // Extra buffer for projections
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
    free(s->xb2);
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


void attention(float* out, float* in, RunState* s, transformerWeights* w, Config* p, int layer, int pos) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = dim / p->n_heads;

    int layer_offset_qkv = layer * dim * dim;
    int layer_offset_kv = layer * kv_dim * dim;

    // 1. QKV Projections
    naive_matmul(s->q, in, w->wq + layer_offset_qkv, dim, dim);
    naive_matmul(s->k, in, w->wk + layer_offset_kv, kv_dim, dim);
    naive_matmul(s->v, in, w->wv + layer_offset_kv, kv_dim, dim);

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
    naive_matmul(out, s->xb, w->wo + layer_offset_qkv, dim, dim);
}

void transformer_block(float* x, RunState* s, transformerWeights* w, Config* p, int layer, int pos) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;

    int layer_offset_ffn = layer * hidden_dim * dim;
    int layer_offset_norm = layer * dim;

    // 1. Attention Block
    // x -> s->xb (normed) -> s->xb2 (attention output)
    RMSNorm(s->xb, x, w->rms_att_weight + layer_offset_norm, dim);
    attention(s->xb2, s->xb, s, w, p, layer, pos);

    for (int i = 0; i < dim; i++) {
        x[i] += s->xb2[i];
    }

    // 2. FeedForward Block
    // x -> s->xb (normed) -> s->xb2 (FFN output)
    RMSNorm(s->xb, x, w->rms_ffn_weight + layer_offset_norm, dim);

    naive_matmul(s->hb, s->xb, w->w1 + layer_offset_ffn, hidden_dim, dim);
    naive_matmul(s->he, s->xb, w->w3 + layer_offset_ffn, hidden_dim, dim);   

    swiglu(s->hb, s->hb, s->he, hidden_dim);

    naive_matmul(s->xb2, s->hb, w->w2 + layer_offset_ffn, dim, hidden_dim);

    for (int i = 0; i < dim; i++) {
        x[i] += s->xb2[i];
    }
}

typedef struct {
    float prob;
    int index;
} ProbIndex;

int compare_prob(const void* a, const void* b) {
    ProbIndex* pa = (ProbIndex*)a;
    ProbIndex* pb = (ProbIndex*)b;
    if (pa->prob > pb->prob) return -1;
    if (pa->prob < pb->prob) return 1;
    return 0;
}

int sample(float* logits, int vocab_size, float temperature, float topp) {
    if (temperature == 0.0f) {
        // Argmax
        int max_idx = 0;
        float max_val = logits[0];
        for (int i = 1; i < vocab_size; i++) {
            if (logits[i] > max_val) {
                max_val = logits[i];
                max_idx = i;
            }
        }
        return max_idx;
    } 
    
    // Apply temperature
    for (int i = 0; i < vocab_size; i++) {
        logits[i] /= temperature;
    }
    
    softmax(logits, vocab_size);

    float r = (float)rand() / (float)RAND_MAX;

    if (topp <= 0.0f || topp >= 1.0f) {
        float cumulative_prob = 0.0f;
        for (int i = 0; i < vocab_size; i++) {
            cumulative_prob += logits[i];
            if (r < cumulative_prob) {
                return i;
            }
        }
        return vocab_size - 1;
    } 
    
    else {
        // Top-p (Nucleus) sampling
        ProbIndex* prob_indices = (ProbIndex*)malloc(vocab_size * sizeof(ProbIndex));
        for (int i = 0; i < vocab_size; i++) {
            prob_indices[i].index = i;
            prob_indices[i].prob = logits[i];
        }
        
        qsort(prob_indices, vocab_size, sizeof(ProbIndex), compare_prob);
        
        float cumulative_prob = 0.0f;
        int last_idx = vocab_size - 1;
        for (int i = 0; i < vocab_size; i++) {
            cumulative_prob += prob_indices[i].prob;
            if (cumulative_prob > topp) {
                last_idx = i;
                break;
            }
        }
        
        float r_scaled = r * cumulative_prob; 
        float current_prob = 0.0f;
        for (int i = 0; i <= last_idx; i++) {
            current_prob += prob_indices[i].prob;
            if (r_scaled < current_prob) {
                int res = prob_indices[i].index;
                free(prob_indices);
                return res;
            }
        }
        
        int res = prob_indices[last_idx].index;
        free(prob_indices);
        return res;
    }
}

int forward(int token, int pos, RunState* s, transformerWeights* w, Config* p, float temperature, float topp) {
    int dim = p->dim;
    
    float* content_row = w->token_embedding_table + token * dim;
    memcpy(s->x, content_row, dim * sizeof(float));

    for (int layer = 0; layer < p->n_layers; layer++) {
        transformer_block(s->x, s, w, p, layer, pos);
    }

    RMSNorm(s->x, s->x, w->rms_final_weight, dim);

    naive_matmul(s->logits, s->x, w->w_cls, p->vocab_size, dim);
    
    return sample(s->logits, p->vocab_size, temperature, topp);
}