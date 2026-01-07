#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"

// Hyperparameters for the Llama architecture
struct Config {
    int dim;        // Transformer dimension (e.g. 288)
    int hidden_dim; // FFN hidden dimension (e.g. 768)
    int n_layers;   // Number of layers
    int n_heads;    // Number of query heads
    int n_kv_heads; // Number of key/value heads (can be < n_heads for GQA)
    int vocab_size; // Vocabulary size (e.g. 32000)
    int seq_len;    // Maximum sequence length (context window)
};

// Storage for model weights (read-only during inference)
struct transformerWeights {
    // Embedding
    float* token_embedding_table; // (vocab_size, dim)

    // Attention Block
    float* rms_att_weight; // (dim) RMSNorm weights for attention input
    float* wq; // (n_heads * head_size, dim) Query projection
    float* wk; // (n_kv_heads * head_size, dim) Key projection
    float* wv; // (n_kv_heads * head_size, dim) Value projection
    float* wo; // (dim, n_heads * head_size) Output projection

    // FeedForward Block
    float* rms_ffn_weight; // (dim) RMSNorm weights for FFN input
    float* w1; // (hidden_dim, dim) Gate (SwiGLU)
    float* w2; // (dim, hidden_dim) Down projection
    float* w3; // (hidden_dim, dim) Up projection

    // Final Output
    float* rms_final_weight; // (dim) Final RMSNorm
    float* w_cls; // (vocab_size, dim) Classifier weights (usually untied)
};

// Runtime state storage (activation buffers and KV cache)
struct RunState {
    // Current state buffers
    float *x;      // Activation at current layer (dim)
    float *xb;     // Buffer for attention/ffn output (dim)
    float *hb;     // Buffer for hidden state (hidden_dim)
    float *he;     // Buffer for hidden state (hidden_dim) - sometimes needed
    float *q;      // Query vector (dim)
    float *k;      // Key vector (dim)
    float *v;      // Value vector (dim)
    float *att;    // Attention scores (n_heads, seq_len)
    float *logits; // Output logits (vocab_size)

    // KV Cache
    // Shape: (n_layers, kv_heads, seq_len, head_size)
    float *key_cache;   
    float *value_cache;
};

void malloc_run_state(RunState* s, Config* p);
void free_run_state(RunState* s);

// Attention Block
// Parameters:
//  x: Input/Output activation vector (size dim)
//  s: Runtime state 
//  w: Model weights
//  p: Model configuration
//  layer: Current layer index (0 to n_layers-1)
//  pos: Current token position index in the sequence
void attention(float* x, RunState* s, transformerWeights* w, Config* p, int layer, int pos);

// Transformer Block (Layer)
// Parameters:
//  x: Input/Output activation vector (size dim)
//  s: Runtime state
//  w: Weights
//  p: Config
//  layer: Layer index
//  pos: Token position
void transformer_block(float* x, RunState* s, transformerWeights* w, Config* p, int layer, int pos);

#endif