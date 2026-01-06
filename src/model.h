#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"

struct Config {

    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

struct transformerWeights {
    float* token_embedding_table;
    float* rms_att_weight;

    float* wq;
    float* wk;
    float* wv;
    float* wo;

    float* rms_ffn_weight;
    float* w1;
    float* w2;
    float* w3;

    float* rms_final_weight;
    float* w_cls;
};

struct RunState {
    // Current state buffers
    float *x;      
    float *xb;     
    float *hb;     
    float *he;     
    float *q;      
    float *k;      
    float *v;      
    float *att;    
    float *logits; 

    // KV Cache
    float *key_cache;
    float *value_cache;
};

void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);

#endif