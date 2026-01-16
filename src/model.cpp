#include "model.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#ifdef USE_CUDA
#include "kernels.cuh"
#endif

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

    //LOGITS - Use pinned memory for faster GPU->CPU transfer during inference
    #ifdef USE_CUDA
    cudaHostAlloc(&s->logits, p->vocab_size * sizeof(float), cudaHostAllocDefault);
    #else
    s->logits = (float*)calloc(p->vocab_size, sizeof(float));
    #endif

    // KV CACHE
    int kv_cache_size = p->n_layers * p->seq_len * kv_dim;
    
    s->key_cache = (float*)calloc(kv_cache_size, sizeof(float));
    s->value_cache = (float*)calloc(kv_cache_size, sizeof(float));

    if (!s->x || !s->key_cache) {
        printf("Malloc failed! System out of memory?\n");
        exit(1);
    }

    #ifdef USE_CUDA
    cudaMalloc(&s->d_x, dim * sizeof(float));
    cudaMalloc(&s->d_xb, dim * sizeof(float));
    cudaMalloc(&s->d_xb2, dim * sizeof(float));
    cudaMalloc(&s->d_hb, hidden_dim * sizeof(float));
    cudaMalloc(&s->d_he, hidden_dim * sizeof(float));
    cudaMalloc(&s->d_q, dim * sizeof(float));
    cudaMalloc(&s->d_k, kv_dim * sizeof(float));
    cudaMalloc(&s->d_v, kv_dim * sizeof(float));
    cudaMalloc(&s->d_att, p->n_heads * p->seq_len * sizeof(float));
    cudaMalloc(&s->d_logits, p->vocab_size * sizeof(float));

    cudaMalloc(&s->d_key_cache, kv_cache_size * sizeof(float));
    cudaMemset(s->d_key_cache, 0, kv_cache_size * sizeof(float));
    cudaMalloc(&s->d_value_cache, kv_cache_size * sizeof(float));
    cudaMemset(s->d_value_cache, 0, kv_cache_size * sizeof(float));
    #endif
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
    #ifdef USE_CUDA
    cudaFreeHost(s->logits);  // Pinned memory needs cudaFreeHost
    #else
    free(s->logits);
    #endif
    free(s->key_cache);
    free(s->value_cache);

    #ifdef USE_CUDA
    cudaFree(s->d_x);
    cudaFree(s->d_xb);
    cudaFree(s->d_xb2);
    cudaFree(s->d_hb);
    cudaFree(s->d_he);
    cudaFree(s->d_q);
    cudaFree(s->d_k);
    cudaFree(s->d_v);
    cudaFree(s->d_att);
    cudaFree(s->d_logits);
    cudaFree(s->d_key_cache);
    cudaFree(s->d_value_cache);
    #endif
}


void attention(RunState* s, transformerWeights* w, Config* p, int layer, int pos) {
    int dim = p->dim;
    int kv_dim = (p->dim * p->n_kv_heads) / p->n_heads;
    int head_size = dim / p->n_heads;
    int cache_stride = p->seq_len * head_size;

    int layer_offset_qkv = layer * dim * dim;
    int layer_offset_kv = layer * kv_dim * dim;
    int gqa_factor = p->n_heads / p->n_kv_heads;

#ifdef USE_CUDA
    // ========== CUDA PATH ==========
    
    cuda_gemv(s->d_q, s->d_xb, w->d_wq + layer_offset_qkv, dim, dim);
    cuda_gemv(s->d_k, s->d_xb, w->d_wk + layer_offset_kv, kv_dim, dim);
    cuda_gemv(s->d_v, s->d_xb, w->d_wv + layer_offset_kv, kv_dim, dim);

    // RoPE on GPU
    cuda_rope(s->d_q, s->d_k, pos, dim, kv_dim, head_size, p->rope_base);
    
    // KV Cache Update - single kernel replaces per-head cudaMemcpy loop
    cuda_scatter_kv(s->d_key_cache, s->d_value_cache, s->d_k, s->d_v, layer, pos, p->n_kv_heads, head_size, p->seq_len);



    // Multi-head attention 
    cuda_multihead_gemv(s->d_att, s->d_q, s->d_key_cache, layer, pos, p->n_heads, p->n_kv_heads, head_size, p->seq_len);
    
    // Scale
    float scale_factor = 1.0f / sqrtf(head_size);
    cuda_scale_multihead(s->d_att, scale_factor, p->n_heads, pos + 1, p->seq_len);
    
    // Softmax 
    cuda_softmax_multihead(s->d_att, s->d_att, p->n_heads, pos + 1, p->seq_len);

    // Aggregation
    int layer_v_offset = layer * (kv_dim * p->seq_len);
    cuda_aggregation_multihead(s->d_xb, s->d_value_cache + layer_v_offset, s->d_att, p->n_heads, pos + 1, head_size, gqa_factor, p->seq_len);

    // Output Projection (result stays in d_xb2 on device)
    cuda_gemv(s->d_xb2, s->d_xb, w->d_wo + layer_offset_qkv, dim, dim);

#else
    // ========== CPU PATH ==========
    naive_matmul(s->q, s->xb, w->wq + layer_offset_qkv, dim, dim);
    naive_matmul(s->k, s->xb, w->wk + layer_offset_kv, kv_dim, dim);
    naive_matmul(s->v, s->xb, w->wv + layer_offset_kv, kv_dim, dim);

    rope(s->q, s->k, pos, dim, kv_dim, head_size, p->rope_base);
    
    for (int h = 0; h < p->n_kv_heads; h++) {
        int src_offset = h * head_size;
        int dst_offset = layer * (kv_dim * p->seq_len) + h * cache_stride + pos * head_size;
        memcpy(s->key_cache + dst_offset, s->k + src_offset, head_size * sizeof(float));
        memcpy(s->value_cache + dst_offset, s->v + src_offset, head_size * sizeof(float));
    }

    for (int h = 0; h < p->n_heads; h++) {
        float* q_head = s->q + h * head_size;
        int kv_head = h / gqa_factor;
        int head_offset = layer * (kv_dim * p->seq_len) + kv_head * cache_stride;
        float* k_cache_head = s->key_cache + head_offset;
        float* v_cache_head = s->value_cache + head_offset;
        float* att_head = s->att + h * p->seq_len;

        naive_matmul(att_head, q_head, k_cache_head, pos + 1, head_size);

        float scale_factor = 1.0f / sqrtf(head_size);
        for (int t = 0; t <= pos; t++) att_head[t] *= scale_factor;
        softmax(att_head, pos + 1);

        float* xb_head = s->xb + h * head_size;
        memset(xb_head, 0, head_size * sizeof(float));
        for (int t = 0; t <= pos; t++) {
            float score = att_head[t];
            float* v_vec = v_cache_head + t * head_size;
            for (int i = 0; i < head_size; i++) xb_head[i] += v_vec[i] * score;
        }
    }

    naive_matmul(s->xb2, s->xb, w->wo + layer_offset_qkv, dim, dim);
#endif
}

void transformer_block(RunState* s, transformerWeights* w, Config* p, int layer, int pos) {
    int dim = p->dim;
    int hidden_dim = p->hidden_dim;

    int layer_offset_ffn = layer * hidden_dim * dim;
    int layer_offset_norm = layer * dim;

#ifdef USE_CUDA
    // ========== CUDA PATH ==========
    
    // Attention block: d_x = d_x + attention(RMSNorm(d_x))
    cuda_rmsnorm(s->d_xb, s->d_x, w->d_rms_att_weight + layer_offset_norm, dim);
    attention(s, w, p, layer, pos);  // Result in d_xb2
    cuda_residual_add(s->d_x, s->d_x, s->d_xb2, dim);
    
    // FFN block: d_x = d_x + FFN(RMSNorm(d_x))
    cuda_rmsnorm(s->d_xb, s->d_x, w->d_rms_ffn_weight + layer_offset_norm, dim);
    cuda_gemv(s->d_hb, s->d_xb, w->d_w1 + layer_offset_ffn, hidden_dim, dim);
    cuda_gemv(s->d_he, s->d_xb, w->d_w3 + layer_offset_ffn, hidden_dim, dim);   
    cuda_swiglu(s->d_hb, s->d_hb, s->d_he, hidden_dim);
    cuda_gemv(s->d_xb2, s->d_hb, w->d_w2 + layer_offset_ffn, dim, hidden_dim);
    cuda_residual_add(s->d_x, s->d_x, s->d_xb2, dim);

#else
    // ========== CPU PATH ==========
    RMSNorm(s->xb, s->x, w->rms_att_weight + layer_offset_norm, dim);
    attention(s, w, p, layer, pos);
    for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
    
    RMSNorm(s->xb, s->x, w->rms_ffn_weight + layer_offset_norm, dim);
    naive_matmul(s->hb, s->xb, w->w1 + layer_offset_ffn, hidden_dim, dim);
    naive_matmul(s->he, s->xb, w->w3 + layer_offset_ffn, hidden_dim, dim);   
    swiglu(s->hb, s->hb, s->he, hidden_dim);
    naive_matmul(s->xb2, s->hb, w->w2 + layer_offset_ffn, dim, hidden_dim);
    for (int i = 0; i < dim; i++) s->x[i] += s->xb2[i];
#endif
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

int sample(float* logits, int vocab_size, float temperature, float topp, 
           float penalty, int* history, int history_len) {
    
    // Apply repetition penalty
    if (penalty > 1.0f && history && history_len > 0) {
        for (int i = 0; i < history_len; i++) {
            int token = history[i];
            if (token >= 0 && token < vocab_size) {
                 if (logits[token] > 0) {
                     logits[token] /= penalty;
                 } else {
                     logits[token] *= penalty;
                 }
            }
        }
    }

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

int forward(int token, int pos, RunState* s, transformerWeights* w, Config* p, float temperature, float topp, float penalty, int* history, int history_len) {
    int dim = p->dim;
    
    float* content_row = w->token_embedding_table + token * dim;

#ifdef USE_CUDA
    // Copy embedding to GPU ONCE at the start
    cudaMemcpy(s->d_x, content_row, dim * sizeof(float), cudaMemcpyHostToDevice);
    
    // All transformer blocks run entirely on GPU
    for (int layer = 0; layer < p->n_layers; layer++) {
        transformer_block(s, w, p, layer, pos);
    }
    
    // Final norm and classifier 
    cuda_rmsnorm(s->d_x, s->d_x, w->d_rms_final_weight, dim);
    cuda_gemv(s->d_logits, s->d_x, w->d_w_cls, p->vocab_size, dim);
    
    // Copy logits back to CPU ONCE at the end
    cudaMemcpy(s->logits, s->d_logits, p->vocab_size * sizeof(float), cudaMemcpyDeviceToHost);
#else
    memcpy(s->x, content_row, dim * sizeof(float));
    
    for (int layer = 0; layer < p->n_layers; layer++) {
        transformer_block(s, w, p, layer, pos);
    }
    
    RMSNorm(s->x, s->x, w->rms_final_weight, dim);
    naive_matmul(s->logits, s->x, w->w_cls, p->vocab_size, dim);
#endif
    
    return sample(s->logits, p->vocab_size, temperature, topp, penalty, history, history_len);
}