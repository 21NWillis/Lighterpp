#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "model.h"
#include "loader.h"
#include "tensor.h"
#include "ops.h"
#include "tokenizer.h"
#include <chrono>
#ifdef USE_CUDA
#include "kernels.cuh"
#include <cuda_runtime.h>
#endif

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path> [tokenizer_path] [temperature] [-p prompt]\n", argv[0]);
        printf("  For GGUF files, tokenizer is embedded (tokenizer_path optional)\n");
        printf("  For .bin files, tokenizer_path is required\n");
        return 1;
    }
    
    float temperature = 0.9f;
    float topp = 0.9f;
    char* prompt = NULL;
    
    // Parse flags (simple scan)
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-p") == 0 && i + 1 < argc) {
            prompt = argv[i + 1];
        }
        if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
             temperature = atof(argv[i+1]);
        }
    }
    
    srand(time(NULL));

    Config config;
    transformerWeights weights;
    RunState state;
    Tokenizer tokenizer;
    memset(&tokenizer, 0, sizeof(Tokenizer));
    
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
        

        if (argc >= 3 && argv[2][0] != '-') {

            if (tokenizer.vocab) free_tokenizer(&tokenizer);
            printf("Loading external tokenizer: %s\n", argv[2]);
            load_tokenizer(&tokenizer, argv[2], config.vocab_size);
        }
        
        if (argc >= 4 && argv[3][0] != '-' && argv[2][0] != '-') temperature = (float)atof(argv[3]);
        
    } else {
        // ===================== llama2.c BIN PATH =====================
        if (argc < 3 || argv[2][0] == '-') {
            printf("Error: .bin files require tokenizer path as 2nd argument\n");
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
        
        if (argc >= 4 && argv[3][0] != '-') temperature = (float)atof(argv[3]);
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
    if (prompt) printf("Prompt: \"%s\"\n", prompt);
    
    #ifdef USE_CUDA
    if (config.hidden_dim > MAX_SHARED_FLOATS) {
        printf("\nError: Model hidden_dim (%d) exceeds CUDA kernel shared memory limit (%d)\n", config.hidden_dim, MAX_SHARED_FLOATS);
        printf("Solution: Increase MAX_SHARED_FLOATS in kernels.cuh and recompile.\n");
        return 1;
    }
    #endif
    
    printf("\n--- Running Inference ---\n");
    
    // Tokenize prompt
    const int MAX_PROMPT_TOKENS = 2048; 
    int prompt_tokens[2048];
    int num_prompt_tokens = 0;
    
    if (prompt) { // use the prompt
        encode(&tokenizer, prompt, prompt_tokens, &num_prompt_tokens, MAX_PROMPT_TOKENS);
        printf("Encoded %d tokens.\n", num_prompt_tokens);
    } else {
        // Default to BOS
        prompt_tokens[0] = 1; 
        num_prompt_tokens = 1;
    }

    int token = prompt_tokens[0]; 
    int pos = 0;
    int steps = 200; 
    
    // Repetition penalty history buffer
    const int HISTORY_LEN = 64;
    int history[64];
    int current_history_len = 0;
    
    // 1. Prefill
    for (int i = 0; i < num_prompt_tokens - 1; i++) {
        // Add to history
        if (current_history_len < HISTORY_LEN) {
            history[current_history_len++] = prompt_tokens[i];
        } else {
            memmove(history, history + 1, (HISTORY_LEN - 1) * sizeof(int));
            history[HISTORY_LEN - 1] = prompt_tokens[i];
        }

        int next_token = forward(prompt_tokens[i], pos, &state, &weights, &config, temperature, topp, 1.1f, history, current_history_len);
        (void)next_token;
        
        char* token_str = decode_token(&tokenizer, prompt_tokens[i]);
        printf("%s", token_str);
        
        pos++;
    }
    
    // 2. Setup for generation
    token = prompt_tokens[num_prompt_tokens - 1]; // Input is last prompt token
    
    // Add last prompt token to history
    if (current_history_len < HISTORY_LEN) {
        history[current_history_len++] = token;
    } else {
        memmove(history, history + 1, (HISTORY_LEN - 1) * sizeof(int));
        history[HISTORY_LEN - 1] = token;
    }
    
    char* token_str = decode_token(&tokenizer, token); // Print it
    printf("%s", token_str);
    fflush(stdout);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 3. Generation loop
    for (; pos < steps; pos++) {
        token = forward(token, pos, &state, &weights, &config, temperature, topp, 1.05f, history, current_history_len);
        
        // Add generated token to history
        if (current_history_len < HISTORY_LEN) {
            history[current_history_len++] = token;
        } else {
            memmove(history, history + 1, (HISTORY_LEN - 1) * sizeof(int));
            history[HISTORY_LEN - 1] = token;
        }
        
        char* token_str = decode_token(&tokenizer, token);
        if (token == 2 || strcmp(token_str, "</s>") == 0 || 
            token == 128001 || strcmp(token_str, "<|end_of_text|>") == 0 ||
            token == 128009 || strcmp(token_str, "<|eot_id|>") == 0) { 
             break;
        }
        
        if (strcmp(token_str, "<0x0A>") == 0) {
            printf("\n");
        } else {
            printf("%s", token_str);
        }
        fflush(stdout);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    
    printf("\n\n--- Statistics ---\n");
    int generated_count = pos - num_prompt_tokens;
    printf("Tokens generated: %d\n", generated_count);
    printf("Elapsed time: %.2f s\n", elapsed.count());
    printf("Tokens per second: %.2f tok/s\n", (float)generated_count / elapsed.count());

    #ifdef USE_CUDA
    free_weights_cuda(&weights);
    #endif
    
    if (tokenizer.vocab) {
        free_tokenizer(&tokenizer);
    }
    
    if (using_gguf) {
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