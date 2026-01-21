#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"

// Weight precision types for kernel dispatch
// Extensible for future quantization (INT4, etc.)
enum WeightPrecision : int {
    WEIGHT_FP32 = 0,   // Standard 32-bit float
    WEIGHT_FP16 = 1,   // Half precision (native on GPU)
    WEIGHT_INT4 = 2    // Future: 4-bit quantized
};

#ifdef USE_CUDA
#include "kernels.cuh"
#include <cuda_runtime.h>
#endif

// Hyperparameters for the Llama architecture
struct Config {
    int dim;        // Transformer dimension (e.g. 288)
    int hidden_dim; // FFN hidden dimension (e.g. 768)
    int n_layers;   // Number of layers
    int n_heads;    // Number of query heads
    int n_kv_heads; // Number of key/value heads (can be < n_heads for GQA)
    int vocab_size; // Vocabulary size (e.g. 32000)
    int seq_len;    // Maximum sequence length (context window)
    float rope_base; // RoPE frequency base (10000 for LLaMA 2, 500000 for LLaMA 3)
    WeightPrecision weight_precision; // Precision of loaded weights
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

    //CUDA device pointers
    // Weight matrices use void* - cast to float* or __half* based on precision
    // Embeddings/norms stay float* (always FP32)
    #ifdef USE_CUDA
    float* d_token_embedding_table;  // Always FP32 (embeddings)

    float* d_rms_att_weight;         // Always FP32 (norm weights are small)
    void* d_wq;                      // Cast based on weight_precision
    void* d_wk;
    void* d_wv;
    void* d_wo;

    float* d_rms_ffn_weight;         // Always FP32
    void* d_w1;
    void* d_w2;
    void* d_w3;

    float* d_rms_final_weight;       // Always FP32
    void* d_w_cls;
    #endif
};

// Runtime state storage (activation buffers and KV cache)
typedef struct {
    // Current state buffers
    float *x;      // Activation at current layer (dim)
    float *xb;     // Buffer for attention/ffn output (dim)
    float *xb2;    // Second buffer to avoid in-place matmul issues (dim)
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

    #ifdef USE_CUDA
    float *d_x;
    float *d_xb;
    float *d_xb2;
    float *d_hb;
    float *d_he;
    float *d_q;
    float *d_k;
    float *d_v;
    float *d_att;
    float *d_logits;

    float *d_key_cache;
    float *d_value_cache;
    void* d_workspace_f16; // Scratch buffer for FP32->FP16 conversions
    
    int* d_sampled_token;  // [1] output from cuda_sample
    int* d_history;        // [seq_len] for repetition penalty
    #endif
} RunState;

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
void attention(RunState* s, transformerWeights* w, Config* p, int layer, int pos);

// Transformer Block (Layer)
// Parameters:
//  x: Input/Output activation vector (size dim)
//  s: Runtime state
//  w: Weights
//  p: Config
//  layer: Layer index
//  pos: Token position
void transformer_block(RunState* s, transformerWeights* w, Config* p, int layer, int pos);

// Sample from logits
// Parameters:
//  logits: Logits vector (size vocab_size)
//  vocab_size: Vocabulary size
//  temperature: Randomness (0.0 = argmax, 1.0 = neutral)
//  topp: Nucleus sampling threshold (0.0 to 1.0, 1.0 = off)
// Returns:
//  Selected token index
int sample(float* logits, int vocab_size, float temperature, float topp);

// Forward Pass
// Parameters:
//  token: Input token ID
//  pos: Token position
//  s: RunState
//  w: Weights
//  p: Config
//  temperature: Sampling temperature
//  topp: Top-p threshold
// Returns:
//  Next token
// penalty: scale factor for repetition penalty (1.0 = no penalty, > 1.0 = penalize)
// history: array of recent token IDs to penalize
// history_len: number of tokens in history array
int forward(int token, int pos, RunState* s, transformerWeights* w, Config* p, float temperature, float topp, float penalty, int* history, int history_len);

#endif