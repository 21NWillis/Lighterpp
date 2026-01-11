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