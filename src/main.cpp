#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "model.h"
#include "loader.h"
#include "tensor.h"
#include "ops.h"
#include "tokenizer.h"
#ifdef USE_CUDA
#include "kernels.cuh"
#include <cuda_runtime.h>
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path> [tokenizer_path] [temperature]\n", argv[0]);
        printf("  For GGUF files, tokenizer is embedded (tokenizer_path optional)\n");
        printf("  For .bin files, tokenizer_path is required\n");
        return 1;
    }
    
    float temperature = 0.9f;
    float topp = 0.9f;
    srand(time(NULL));

    Config config;
    transformerWeights weights;
    RunState state;
    Tokenizer tokenizer;
    
    // For cleanup tracking
    GGUFFile* gguf = nullptr;
    float* bin_data = nullptr;
    size_t bin_file_size = 0;
    bool using_gguf = false;

    printf("Loading model: %s\n", argv[1]);
    
    if (is_gguf_file(argv[1])) {
        // ===================== GGUF PATH =====================
        using_gguf = true;
        
        gguf = gguf_open(argv[1]);
        if (!gguf) {
            printf("Error: Failed to open GGUF file.\n");
            return 1;
        }
        
        if (!gguf_extract_config(gguf, &config)) {
            printf("Error: Failed to extract config from GGUF.\n");
            gguf_close(gguf);
            return 1;
        }
        
        memset(&weights, 0, sizeof(weights));
        if (!gguf_init_weights(gguf, &weights, &config)) {
            printf("Error: Failed to load weights from GGUF.\n");
            gguf_close(gguf);
            return 1;
        }
        
        // Load tokenizer from GGUF (embedded vocabulary)
        if (!gguf_init_tokenizer(gguf, &tokenizer)) {
            printf("Warning: Could not load tokenizer from GGUF, output will be token IDs\n");
            tokenizer.vocab = nullptr;
            tokenizer.vocab_size = config.vocab_size;
        }
        
        // Optional: external tokenizer override
        if (argc >= 3 && strcmp(argv[2], "-") != 0) {
            // Free GGUF tokenizer if loaded, use external instead
            if (tokenizer.vocab) free_tokenizer(&tokenizer);
            printf("Loading external tokenizer: %s\n", argv[2]);
            load_tokenizer(&tokenizer, argv[2], config.vocab_size);
        }
        
        if (argc >= 4) temperature = (float)atof(argv[3]);
        
    } else {
        // ===================== llama2.c BIN PATH =====================
        if (argc < 3) {
            printf("Error: .bin files require tokenizer path\n");
            printf("Usage: %s <model.bin> <tokenizer.bin> [temperature]\n", argv[0]);
            return 1;
        }
        
        bin_data = load_model_file(argv[1], &config, &bin_file_size);
        if (!bin_data) {
            printf("Error: Failed to load model.\n");
            return 1;
        }
        
        printf("Loading tokenizer: %s\n", argv[2]);
        load_tokenizer(&tokenizer, argv[2], config.vocab_size);
        
        checkpoint_init_weights(&weights, &config, bin_data);
        
        if (argc >= 4) temperature = (float)atof(argv[3]);
    }

    malloc_run_state(&state, &config);

    printf("\n--- Model Configuration ---\n");
    printf("Dim: %d\n", config.dim);
    printf("Hidden Dim: %d\n", config.hidden_dim);
    printf("Layers: %d\n", config.n_layers);
    printf("Heads: %d\n", config.n_heads);
    printf("KV Heads: %d\n", config.n_kv_heads);
    printf("Vocab: %d\n", config.vocab_size);
    printf("Seq Len: %d\n", config.seq_len);
    printf("RoPE Base: %.0f\n", config.rope_base);
    printf("Temperature: %.2f\n", temperature);
    
    #ifdef USE_CUDA
    if (config.hidden_dim > MAX_SHARED_FLOATS) {
        printf("\nError: Model hidden_dim (%d) exceeds CUDA kernel shared memory limit (%d)\n", config.hidden_dim, MAX_SHARED_FLOATS);
        printf("Solution: Increase MAX_SHARED_FLOATS in kernels.cuh and recompile.\n");
        return 1;
    }
    #endif
    
    printf("\n--- Running Inference ---\n");
    
    int token = 1;  // BOS token
    int steps = 256;
    clock_t start = clock();

    for (int pos = 0; pos < steps; pos++) {
        token = forward(token, pos, &state, &weights, &config, temperature, topp);
        
        if (tokenizer.vocab) {
            char* token_str = decode_token(&tokenizer, token);
            if (strcmp(token_str, "<0x0A>") == 0) {
                printf("\n");
            } else {
                printf("%s", token_str);
            }
        } else {
            printf("[%d]", token);
        }
        fflush(stdout);
    }
    
    clock_t end = clock();
    double elapsed = (double)(end - start) / CLOCKS_PER_SEC;
    
    printf("\n\n--- Statistics ---\n");
    printf("Tokens generated: %d\n", steps);
    printf("Elapsed time: %.2f s\n", elapsed);
    printf("Tokens per second: %.2f tok/s\n", (float)steps / elapsed);

    #ifdef USE_CUDA
    free_weights_cuda(&weights);
    #endif
    
    if (tokenizer.vocab) {
        free_tokenizer(&tokenizer);
    }
    
    if (using_gguf) {
        // GGUF weights were malloc'd, need to free individually
        free(weights.token_embedding_table);
        if (weights.w_cls != weights.token_embedding_table) free(weights.w_cls);
        free(weights.rms_att_weight);
        free(weights.wq);
        free(weights.wk);
        free(weights.wv);
        free(weights.wo);
        free(weights.rms_ffn_weight);
        free(weights.w1);
        free(weights.w2);
        free(weights.w3);
        free(weights.rms_final_weight);
        gguf_close(gguf);
    } else {
        free_model_file(bin_data, bin_file_size);
    }
    
    free_run_state(&state);
    printf("\nModel Unloaded. Exiting.\n");

    return 0;
}