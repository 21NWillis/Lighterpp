#include "model.h"
#include <stdlib.h>
#include <stdio.h>

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