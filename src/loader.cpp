#include "loader.h"
#include "model.h"
#include "tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <cstring>

#ifdef USE_CUDA
#include "kernels.cuh"
#include <cuda_runtime.h>
#endif

void checkpoint_init_weights(transformerWeights* w, Config* p, float* ptr) {
    int head_size = p->dim/ p->n_heads;

    // Token Embedding Table: (vocab_size, dim)
    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;

    // Attention RMSNorm weights: (n_layers, dim)
    w->rms_att_weight = ptr;
    ptr += p->n_layers * p->dim;

    // Query projection: (n_layers, dim, n_heads * head_size)
    w->wq = ptr;
    ptr += p->n_layers * p->dim * (p->n_heads * head_size);

    // Key projection: (n_layers, dim, n_kv_heads * head_size)
    w->wk = ptr;
    ptr += p->n_layers * p->dim * (p->n_kv_heads * head_size);

    // Value projection: (n_layers, dim, n_kv_heads * head_size)
    w->wv = ptr;
    ptr += p->n_layers * p->dim * (p->n_kv_heads * head_size);

    // Output projection: (n_layers, n_heads * head_size, dim)
    w->wo = ptr;
    ptr += p->n_layers * (p->n_heads * head_size) * p->dim;

    // FeedForward RMSNorm weights: (n_layers, dim)
    w->rms_ffn_weight = ptr;
    ptr += p->n_layers * p->dim;

    // FFN Gate projection (w1): (n_layers, dim, hidden_dim)
    w->w1 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;

    // FFN Down projection (w2): (n_layers, hidden_dim, dim)
    w->w2 = ptr;
    ptr += p->n_layers * p->hidden_dim * p->dim;

    // FFN Up projection (w3): (n_layers, dim, hidden_dim)
    w->w3 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;

    // Final RMSNorm weights: (dim)
    w->rms_final_weight = ptr;
    ptr += p->dim;

    // w_cls (Classifier / Un-embedding)
    w->w_cls = w->token_embedding_table;

    // =========================================================================
    // CUDA: Allocate GPU memory and copy weights from CPU to GPU
    // Each weight tensor needs: cudaMalloc (allocate) + cudaMemcpy (transfer)
    // Sizes match the pointer arithmetic used above for CPU weights
    // =========================================================================
    #ifdef USE_CUDA
    size_t size;  
    
    // Token Embedding: (vocab_size, dim) - also used as classifier (w_cls)
    size = p->vocab_size * p->dim * sizeof(float);
    cudaMalloc(&w->d_token_embedding_table, size);
    cudaMemcpy(w->d_token_embedding_table, w->token_embedding_table, size, cudaMemcpyHostToDevice);
    w->d_w_cls = w->d_token_embedding_table;  // Tied weights: classifier shares embedding
    
    // Attention RMSNorm: (n_layers, dim)
    size = p->n_layers * p->dim * sizeof(float);
    cudaMalloc(&w->d_rms_att_weight, size);
    cudaMemcpy(w->d_rms_att_weight, w->rms_att_weight, size, cudaMemcpyHostToDevice);
    
    // Query projection: (n_layers, dim, n_heads * head_size) = (n_layers, dim, dim)
    size = p->n_layers * p->dim * (p->n_heads * head_size) * sizeof(float);
    cudaMalloc(&w->d_wq, size);
    cudaMemcpy(w->d_wq, w->wq, size, cudaMemcpyHostToDevice);
    
    // Key projection: (n_layers, dim, n_kv_heads * head_size)
    size = p->n_layers * p->dim * (p->n_kv_heads * head_size) * sizeof(float);
    cudaMalloc(&w->d_wk, size);
    cudaMemcpy(w->d_wk, w->wk, size, cudaMemcpyHostToDevice);
    
    // Value projection: (n_layers, dim, n_kv_heads * head_size)
    size = p->n_layers * p->dim * (p->n_kv_heads * head_size) * sizeof(float);
    cudaMalloc(&w->d_wv, size);
    cudaMemcpy(w->d_wv, w->wv, size, cudaMemcpyHostToDevice);
    
    // Output projection: (n_layers, n_heads * head_size, dim) = (n_layers, dim, dim)
    size = p->n_layers * (p->n_heads * head_size) * p->dim * sizeof(float);
    cudaMalloc(&w->d_wo, size);
    cudaMemcpy(w->d_wo, w->wo, size, cudaMemcpyHostToDevice);
    
    // FFN RMSNorm: (n_layers, dim)
    size = p->n_layers * p->dim * sizeof(float);
    cudaMalloc(&w->d_rms_ffn_weight, size);
    cudaMemcpy(w->d_rms_ffn_weight, w->rms_ffn_weight, size, cudaMemcpyHostToDevice);
    
    // FFN Gate (w1): (n_layers, dim, hidden_dim)
    size = p->n_layers * p->dim * p->hidden_dim * sizeof(float);
    cudaMalloc(&w->d_w1, size);
    cudaMemcpy(w->d_w1, w->w1, size, cudaMemcpyHostToDevice);
    
    // FFN Down (w2): (n_layers, hidden_dim, dim)
    size = p->n_layers * p->hidden_dim * p->dim * sizeof(float);
    cudaMalloc(&w->d_w2, size);
    cudaMemcpy(w->d_w2, w->w2, size, cudaMemcpyHostToDevice);
    
    // FFN Up (w3): (n_layers, dim, hidden_dim)
    size = p->n_layers * p->dim * p->hidden_dim * sizeof(float);
    cudaMalloc(&w->d_w3, size);
    cudaMemcpy(w->d_w3, w->w3, size, cudaMemcpyHostToDevice);
    
    // Final RMSNorm: (dim) - just one layer, not per-layer
    size = p->dim * sizeof(float);
    cudaMalloc(&w->d_rms_final_weight, size);
    cudaMemcpy(w->d_rms_final_weight, w->rms_final_weight, size, cudaMemcpyHostToDevice);
    #endif
}

float* load_model_file(const char* checkpoint_path, Config* config, size_t* file_size_out) {
    
    int file = open(checkpoint_path, O_RDONLY);
    if (file == -1) {
        std::cerr << "Failed to open file: " << checkpoint_path << std::endl;
        return NULL;
    }

    struct stat sb;
    if (fstat(file, &sb) == -1) {
        std::cerr << "Failed to get file size." << std::endl;
        close(file);
        return nullptr;
    }
    size_t file_size = sb.st_size;

    void* addr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, file, 0);
    
    if (addr == MAP_FAILED) {
        std::cerr << "Mmap failed!" << std::endl;
        close(file);
        return nullptr;
    }

    Config* file_config = (Config*)addr;
    *config = *file_config;
    
    // llama2.c format doesn't include rope_base, default to 10000 (LLaMA 2)
    config->rope_base = 10000.0f;
    
    float* weightspointer = (float*)((char*)addr + 28);

    *file_size_out = file_size;

    close(file);

    std::cout << "Model loaded successfully." << std::endl;
    return weightspointer;
}

void free_model_file(float* data, size_t file_size) {
    void* mmap_start = (void*)((char*)data - 28);
    munmap(mmap_start, file_size);
}

// Free GPU weight memory (call before free_model_file)
#ifdef USE_CUDA
void free_weights_cuda(transformerWeights* w) {
    cudaFree(w->d_token_embedding_table);
    // d_w_cls is tied to d_token_embedding_table, don't double-free
    cudaFree(w->d_rms_att_weight);
    cudaFree(w->d_wq);
    cudaFree(w->d_wk);
    cudaFree(w->d_wv);
    cudaFree(w->d_wo);
    cudaFree(w->d_rms_ffn_weight);
    cudaFree(w->d_w1);
    cudaFree(w->d_w2);
    cudaFree(w->d_w3);
    cudaFree(w->d_rms_final_weight);
}
#endif

// =============================================================================
// GGUF FORMAT LOADING
// =============================================================================

bool is_gguf_file(const char* path) {
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    
    uint32_t magic;
    size_t read = fread(&magic, sizeof(uint32_t), 1, f);
    fclose(f);
    
    // GGUF magic is "GGUF" = 0x46554747 in little-endian
    return (read == 1 && magic == 0x46554747);
}

bool gguf_extract_config(GGUFFile* file, Config* config) {
    // Get architecture name to determine key prefix
    // LLaMA models use "llama.*" keys
    char* arch = nullptr;
    if (!gguf_get_string(file, "general.architecture", &arch)) {
        std::cerr << "GGUF: Missing general.architecture" << std::endl;
        return false;
    }
    
    std::cout << "GGUF: Architecture = " << arch << std::endl;
    
    // Build key prefix (e.g., "llama.")
    char prefix[64];
    snprintf(prefix, sizeof(prefix), "%s.", arch);
    
    // Helper to build full key names
    char key[128];
    
    // Extract embedding dimension
    snprintf(key, sizeof(key), "%sembedding_length", prefix);
    uint32_t dim;
    if (!gguf_get_uint32(file, key, &dim)) {
        std::cerr << "GGUF: Missing " << key << std::endl;
        free(arch);
        return false;
    }
    config->dim = dim;
    
    // Extract hidden dimension (FFN intermediate size)
    snprintf(key, sizeof(key), "%sfeed_forward_length", prefix);
    uint32_t hidden_dim;
    if (!gguf_get_uint32(file, key, &hidden_dim)) {
        std::cerr << "GGUF: Missing " << key << std::endl;
        free(arch);
        return false;
    }
    config->hidden_dim = hidden_dim;
    
    // Extract number of layers
    snprintf(key, sizeof(key), "%sblock_count", prefix);
    uint32_t n_layers;
    if (!gguf_get_uint32(file, key, &n_layers)) {
        std::cerr << "GGUF: Missing " << key << std::endl;
        free(arch);
        return false;
    }
    config->n_layers = n_layers;
    
    // Extract number of attention heads
    snprintf(key, sizeof(key), "%sattention.head_count", prefix);
    uint32_t n_heads;
    if (!gguf_get_uint32(file, key, &n_heads)) {
        std::cerr << "GGUF: Missing " << key << std::endl;
        free(arch);
        return false;
    }
    config->n_heads = n_heads;
    
    // Extract number of KV heads (for GQA)
    snprintf(key, sizeof(key), "%sattention.head_count_kv", prefix);
    uint32_t n_kv_heads;
    if (!gguf_get_uint32(file, key, &n_kv_heads)) {
        // If not present, assume same as n_heads (no GQA)
        n_kv_heads = n_heads;
    }
    config->n_kv_heads = n_kv_heads;
    
    // Extract context length
    snprintf(key, sizeof(key), "%scontext_length", prefix);
    uint32_t seq_len;
    if (!gguf_get_uint32(file, key, &seq_len)) {
        std::cerr << "GGUF: Missing " << key << std::endl;
        free(arch);
        return false;
    }
    config->seq_len = seq_len;
    
    // Extract RoPE frequency base (defaults to 10000 for LLaMA 2)
    snprintf(key, sizeof(key), "%srope.freq_base", prefix);
    float rope_base;
    if (!gguf_get_float32(file, key, &rope_base)) {
        rope_base = 10000.0f;  // Default for LLaMA 2
    }
    config->rope_base = rope_base;
    
    // Get vocabulary size from tokenizer
    // We count the tokens array rather than reading a separate field
    char** vocab;
    size_t vocab_size = gguf_get_string_array(file, "tokenizer.ggml.tokens", &vocab);
    if (vocab_size == 0) {
        std::cerr << "GGUF: Missing tokenizer.ggml.tokens" << std::endl;
        free(arch);
        return false;
    }
    config->vocab_size = vocab_size;
    
    // Free vocabulary (we just needed the count)
    for (size_t i = 0; i < vocab_size; i++) free(vocab[i]);
    free(vocab);
    
    free(arch);
    
    std::cout << "GGUF Config: dim=" << config->dim
              << ", hidden_dim=" << config->hidden_dim
              << ", n_layers=" << config->n_layers
              << ", n_heads=" << config->n_heads
              << ", n_kv_heads=" << config->n_kv_heads
              << ", vocab_size=" << config->vocab_size
              << ", seq_len=" << config->seq_len
              << ", rope_base=" << config->rope_base
              << std::endl;
    
    // VRAM Safety: Clamp sequence length for consumer GPUs
    // Llama 3 defaults to 131k, which requires ~4GB VRAM just for KV cache.
    // We cap it to 16k (~500MB) to be safe on 8GB cards.
    if (config->seq_len > 16384) {
        std::cout << "GGUF Warning: Model requests " << config->seq_len << " seq_len. Clamping to 16384 to prevent VRAM overflow." << std::endl;
        config->seq_len = 16384;
    }

    return true;
}

// =============================================================================
// GGUF Weight Loading
// 
// Key differences from llama2.c format:
// 1. Tensors are named (e.g., "blk.5.attn_q.weight") instead of sequential
// 2. Per-layer tensors must be concatenated into stacked arrays
// 3. May need F16→F32 conversion
// =============================================================================

// Convert F16 to F32 (IEEE 754 half-precision)
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1F;
    uint32_t mant = h & 0x03FF;
    
    uint32_t result;
    float f_result;
    
    if (exp == 0) {
        // Subnormal or zero
        if (mant == 0) {
            result = sign;
            memcpy(&f_result, &result, sizeof(float));
            return f_result;
        }
        // Subnormal: normalize
        while ((mant & 0x0400) == 0) {
            mant <<= 1;
            exp--;
        }
        exp++;
        mant &= ~0x0400;
    } else if (exp == 31) {
        // Inf or NaN
        result = sign | 0x7F800000 | (mant << 13);
        memcpy(&f_result, &result, sizeof(float));
        return f_result;
    }
    
    exp += 127 - 15;  // Adjust exponent bias
    result = sign | (exp << 23) | (mant << 13);
    memcpy(&f_result, &result, sizeof(float));
    return f_result;
}

// Copy tensor data, converting F16 to F32 if needed
static void copy_tensor_data(float* dst, GGUFFile* file, GGUFTensorInfo* tensor) {
    void* src = gguf_tensor_data(file, tensor);
    size_t nelements = 1;
    for (uint32_t d = 0; d < tensor->n_dims; d++) {
        nelements *= tensor->dims[d];
    }
    
    if (tensor->type == GGML_TYPE_F32) {
        memcpy(dst, src, nelements * sizeof(float));
    } else if (tensor->type == GGML_TYPE_F16) {
        uint16_t* src16 = (uint16_t*)src;
        for (size_t i = 0; i < nelements; i++) {
            dst[i] = f16_to_f32(src16[i]);
        }
    } else {
        std::cerr << "GGUF: Unsupported tensor type " << tensor->type 
                  << " for " << tensor->name << std::endl;
        // Zero fill for unsupported types
        memset(dst, 0, nelements * sizeof(float));
    }
}



#ifdef USE_CUDA

// Upload tensor to a specific offset in an existing GPU buffer
// Used for stacking per-layer weights
static void upload_tensor_to_offset(GGUFFile* file, GGUFTensorInfo* tensor, 
                                    void* d_buffer, size_t element_offset, size_t elem_size) {
    void* src = gguf_tensor_data(file, tensor);
    size_t nelements = 1;
    for (uint32_t d = 0; d < tensor->n_dims; d++) {
        nelements *= tensor->dims[d];
    }
    
    size_t byte_offset = element_offset * elem_size;
    size_t byte_size = nelements * elem_size;
    
    cudaMemcpy((char*)d_buffer + byte_offset, src, byte_size, cudaMemcpyHostToDevice);
}
#endif

bool gguf_init_weights(GGUFFile* file, transformerWeights* w, Config* p) {
    int head_size = p->dim / p->n_heads;
    int kv_dim = p->n_kv_heads * head_size;
    
    // Detect weight precision from first weight tensor
    GGUFTensorInfo* sample = gguf_find_tensor(file, "blk.0.attn_q.weight");
    if (sample && sample->type == GGML_TYPE_F16) {
        p->weight_precision = WEIGHT_FP16;
        std::cout << "GGUF: Detected FP16 weights" << std::endl;
    } else {
        p->weight_precision = WEIGHT_FP32;
        std::cout << "GGUF: Detected FP32 weights" << std::endl;
    }
#ifdef USE_CUDA
    size_t elem_size = (p->weight_precision == WEIGHT_FP16) ? 2 : 4;
#endif
    
    // Allocate weight arrays (stacked across all layers)
    w->token_embedding_table = (float*)malloc(p->vocab_size * p->dim * sizeof(float));
    w->rms_att_weight = (float*)malloc(p->n_layers * p->dim * sizeof(float));
    w->wq = (float*)malloc(p->n_layers * p->dim * p->dim * sizeof(float));
    w->wk = (float*)malloc(p->n_layers * p->dim * kv_dim * sizeof(float));
    w->wv = (float*)malloc(p->n_layers * p->dim * kv_dim * sizeof(float));
    w->wo = (float*)malloc(p->n_layers * p->dim * p->dim * sizeof(float));
    w->rms_ffn_weight = (float*)malloc(p->n_layers * p->dim * sizeof(float));
    w->w1 = (float*)malloc(p->n_layers * p->dim * p->hidden_dim * sizeof(float));
    w->w2 = (float*)malloc(p->n_layers * p->hidden_dim * p->dim * sizeof(float));
    w->w3 = (float*)malloc(p->n_layers * p->dim * p->hidden_dim * sizeof(float));
    w->rms_final_weight = (float*)malloc(p->dim * sizeof(float));
    
    // Token embedding
    GGUFTensorInfo* t = gguf_find_tensor(file, "token_embd.weight");
    if (!t) {
        std::cerr << "GGUF: Missing token_embd.weight" << std::endl;
        return false;
    }
    copy_tensor_data(w->token_embedding_table, file, t);
    
    // Output/classifier weights (may be tied to embedding or separate)
    t = gguf_find_tensor(file, "output.weight");
    if (t) {
        // Separate output weights - allocate and copy
        w->w_cls = (float*)malloc(p->vocab_size * p->dim * sizeof(float));
        copy_tensor_data(w->w_cls, file, t);
    } else {
        // Tied weights - share with embedding
        w->w_cls = w->token_embedding_table;
    }
    
    // Final RMSNorm
    t = gguf_find_tensor(file, "output_norm.weight");
    if (!t) {
        std::cerr << "GGUF: Missing output_norm.weight" << std::endl;
        return false;
    }
    copy_tensor_data(w->rms_final_weight, file, t);
    
    // Per-layer weights
    char name[128];
    for (int layer = 0; layer < p->n_layers; layer++) {
        // Attention norm
        snprintf(name, sizeof(name), "blk.%d.attn_norm.weight", layer);
        t = gguf_find_tensor(file, name);
        if (!t) { std::cerr << "GGUF: Missing " << name << std::endl; return false; }
        copy_tensor_data(w->rms_att_weight + layer * p->dim, file, t);
        
        // Q projection
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", layer);
        t = gguf_find_tensor(file, name);
        if (!t) { std::cerr << "GGUF: Missing " << name << std::endl; return false; }
        // GGUF Q/K weights are permuted for RoPE (adjacent pairs 0,1)
        // We un-permute them to standard Llama layout (split pairs 0, n/2)
        copy_tensor_data(w->wq + layer * p->dim * p->dim, file, t);
        
        // K projection
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", layer);
        t = gguf_find_tensor(file, name);
        if (!t) { std::cerr << "GGUF: Missing " << name << std::endl; return false; }
        copy_tensor_data(w->wk + layer * p->dim * kv_dim, file, t);
        
        // V projection
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", layer);
        t = gguf_find_tensor(file, name);
        if (!t) { std::cerr << "GGUF: Missing " << name << std::endl; return false; }
        copy_tensor_data(w->wv + layer * p->dim * kv_dim, file, t);
        
        // Output projection
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", layer);
        t = gguf_find_tensor(file, name);
        if (!t) { std::cerr << "GGUF: Missing " << name << std::endl; return false; }
        copy_tensor_data(w->wo + layer * p->dim * p->dim, file, t);
        
        // FFN norm
        snprintf(name, sizeof(name), "blk.%d.ffn_norm.weight", layer);
        t = gguf_find_tensor(file, name);
        if (!t) { std::cerr << "GGUF: Missing " << name << std::endl; return false; }
        copy_tensor_data(w->rms_ffn_weight + layer * p->dim, file, t);
        
        // FFN gate (w1)
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", layer);
        t = gguf_find_tensor(file, name);
        if (!t) { std::cerr << "GGUF: Missing " << name << std::endl; return false; }
        copy_tensor_data(w->w1 + layer * p->dim * p->hidden_dim, file, t);
        
        // FFN down (w2)
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", layer);
        t = gguf_find_tensor(file, name);
        if (!t) { std::cerr << "GGUF: Missing " << name << std::endl; return false; }
        copy_tensor_data(w->w2 + layer * p->hidden_dim * p->dim, file, t);
        
        // FFN up (w3)
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", layer);
        t = gguf_find_tensor(file, name);
        if (!t) { std::cerr << "GGUF: Missing " << name << std::endl; return false; }
        copy_tensor_data(w->w3 + layer * p->dim * p->hidden_dim, file, t);
    }
    

    #ifdef USE_CUDA
    size_t size;
    
    // Embeddings/norms always FP32
    size = p->vocab_size * p->dim * sizeof(float);
    cudaMalloc(&w->d_token_embedding_table, size);
    cudaMemcpy(w->d_token_embedding_table, w->token_embedding_table, size, cudaMemcpyHostToDevice);

    size = p->n_layers * p->dim * sizeof(float);
    cudaMalloc(&w->d_rms_att_weight, size);
    cudaMemcpy(w->d_rms_att_weight, w->rms_att_weight, size, cudaMemcpyHostToDevice);
    
    size = p->n_layers * p->dim * sizeof(float);
    cudaMalloc(&w->d_rms_ffn_weight, size);
    cudaMemcpy(w->d_rms_ffn_weight, w->rms_ffn_weight, size, cudaMemcpyHostToDevice);
    
    size = p->dim * sizeof(float);
    cudaMalloc(&w->d_rms_final_weight, size);
    cudaMemcpy(w->d_rms_final_weight, w->rms_final_weight, size, cudaMemcpyHostToDevice);

    // Weight matrices: use elem_size (2 for F16, 4 for F32)
    // Allocate GPU buffers for stacked weights
    size = p->n_layers * p->dim * p->dim * elem_size;
    cudaMalloc(&w->d_wq, size);
    cudaMalloc(&w->d_wo, size);
    
    size = p->n_layers * p->dim * kv_dim * elem_size;
    cudaMalloc(&w->d_wk, size);
    cudaMalloc(&w->d_wv, size);
    
    size = p->n_layers * p->dim * p->hidden_dim * elem_size;
    cudaMalloc(&w->d_w1, size);
    cudaMalloc(&w->d_w3, size);
    
    size = p->n_layers * p->hidden_dim * p->dim * elem_size;
    cudaMalloc(&w->d_w2, size);
    
    // Upload per-layer weights directly from GGUF (preserves F16)
    for (int layer = 0; layer < p->n_layers; layer++) {
        snprintf(name, sizeof(name), "blk.%d.attn_q.weight", layer);
        GGUFTensorInfo* t = gguf_find_tensor(file, name);
        upload_tensor_to_offset(file, t, w->d_wq, layer * p->dim * p->dim, elem_size);
        
        snprintf(name, sizeof(name), "blk.%d.attn_k.weight", layer);
        t = gguf_find_tensor(file, name);
        upload_tensor_to_offset(file, t, w->d_wk, layer * p->dim * kv_dim, elem_size);
        
        snprintf(name, sizeof(name), "blk.%d.attn_v.weight", layer);
        t = gguf_find_tensor(file, name);
        upload_tensor_to_offset(file, t, w->d_wv, layer * p->dim * kv_dim, elem_size);
        
        snprintf(name, sizeof(name), "blk.%d.attn_output.weight", layer);
        t = gguf_find_tensor(file, name);
        upload_tensor_to_offset(file, t, w->d_wo, layer * p->dim * p->dim, elem_size);
        
        snprintf(name, sizeof(name), "blk.%d.ffn_gate.weight", layer);
        t = gguf_find_tensor(file, name);
        upload_tensor_to_offset(file, t, w->d_w1, layer * p->dim * p->hidden_dim, elem_size);
        
        snprintf(name, sizeof(name), "blk.%d.ffn_down.weight", layer);
        t = gguf_find_tensor(file, name);
        upload_tensor_to_offset(file, t, w->d_w2, layer * p->hidden_dim * p->dim, elem_size);
        
        snprintf(name, sizeof(name), "blk.%d.ffn_up.weight", layer);
        t = gguf_find_tensor(file, name);
        upload_tensor_to_offset(file, t, w->d_w3, layer * p->dim * p->hidden_dim, elem_size);
    }
    
    // Classifier (output) weights
    GGUFTensorInfo* t_cls = gguf_find_tensor(file, "output.weight");
    if (t_cls) {
        std::cout << "GGUF: Using separate output weights (Type: " << t_cls->type << ")" << std::endl;
        size = p->vocab_size * p->dim * elem_size;
        cudaMalloc(&w->d_w_cls, size);
        void* src = gguf_tensor_data(file, t_cls);
        cudaMemcpy(w->d_w_cls, src, size, cudaMemcpyHostToDevice);
    } else {
        std::cout << "GGUF: Tied weights detected (sharing with embeddings)" << std::endl;
        if (p->weight_precision == WEIGHT_FP16) {
            std::cout << "GGUF: Converting tied FP32 embeddings to FP16 for classifier..." << std::endl;
            size = p->vocab_size * p->dim * sizeof(uint16_t);
            cudaMalloc(&w->d_w_cls, size);
            
            // Declared in kernels.cuh: cuda_convert_f32_to_f16
            cuda_convert_f32_to_f16(w->d_w_cls, w->d_token_embedding_table, p->vocab_size * p->dim);
        } else {

            w->d_w_cls = w->d_token_embedding_table;
        }
    }
    
    std::cout << "GGUF: Copied weights to GPU (" 
              << (p->weight_precision == WEIGHT_FP16 ? "FP16" : "FP32") 
              << ")" << std::endl;
    #endif
    
    std::cout << "GGUF: Loaded weights for " << p->n_layers << " layers" << std::endl;
    return true;
}

// =============================================================================
// GGUF Tokenizer Loading
// Reads vocabulary directly from GGUF metadata (no separate tokenizer file)
// =============================================================================

bool gguf_init_tokenizer(GGUFFile* file, Tokenizer* tokenizer) {
    // - "llama" = SentencePiece (LLaMA 2, uses ▁ for space)
    // - "gpt2"  = Tiktoken BPE (LLaMA 3, uses Ġ for space)
    char* tokenizer_model = nullptr;
    if (gguf_get_string(file, "tokenizer.ggml.model", &tokenizer_model)) {
        if (strcmp(tokenizer_model, "gpt2") == 0) {
            tokenizer->type = TOKENIZER_BPE;
            std::cout << "GGUF: Detected tokenizer type = BPE (LLaMA 3 style)" << std::endl;
        } else {
            tokenizer->type = TOKENIZER_SENTENCEPIECE;
            std::cout << "GGUF: Detected tokenizer type = SentencePiece (" << tokenizer_model << ")" << std::endl;
        }
        free(tokenizer_model);
    } else {
        // Default to SentencePiece if not specified
        tokenizer->type = TOKENIZER_SENTENCEPIECE;
        std::cout << "GGUF: No tokenizer.ggml.model found, defaulting to SentencePiece" << std::endl;
    }
    
    // Get vocabulary tokens
    char** vocab = nullptr;
    size_t vocab_size = gguf_get_string_array(file, "tokenizer.ggml.tokens", &vocab);
    if (vocab_size == 0) {
        std::cerr << "GGUF: No tokenizer.ggml.tokens found" << std::endl;
        return false;
    }
    
    tokenizer->vocab_size = vocab_size;
    tokenizer->vocab = vocab;  // Transfer ownership
    
    // Get scores (optional - some models may not have them)
    float* scores = nullptr;
    size_t score_count = gguf_get_float32_array(file, "tokenizer.ggml.scores", &scores);
    if (score_count == vocab_size) {
        tokenizer->vocab_scores = scores;
    } else {
        // Scores missing or count mismatch - allocate default scores
        tokenizer->vocab_scores = (float*)calloc(vocab_size, sizeof(float));
        if (scores) free(scores);
    }
    
    // Find max token length (for encoding, if needed later)
    int max_len = 0;
    for (size_t i = 0; i < vocab_size; i++) {
        int len = strlen(vocab[i]);
        if (len > max_len) max_len = len;
    }
    tokenizer->max_token_len = max_len;
    
    std::cout << "GGUF: Loaded tokenizer with " << vocab_size << " tokens" << std::endl;
    return true;
}
