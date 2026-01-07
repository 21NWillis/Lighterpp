#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "model.h"
#include "loader.h"
#include "tensor.h"
#include "ops.h"

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    Config config;
    transformerWeights weights;
    RunState state;
    
    size_t file_size = 0;

    printf("Loading model: %s\n", argv[1]);
    
    float* data = load_model_file(argv[1], &config, &file_size);
    if (!data) {
        printf("Error: Failed to load model.\n");
        return 1;
    }

    malloc_run_state(&state, &config);

    int kv_dim = (config.dim * config.n_kv_heads) / config.n_heads;
    size_t cache_elements = config.n_layers * config.seq_len * kv_dim;
    size_t cache_bytes = cache_elements * sizeof(float);

    printf("KV Cache Size: %zu bytes\n", cache_bytes);

    checkpoint_init_weights(&weights, &config, data);

    printf("\n--- Model Configuration ---\n");
    printf("Dim: %d\n", config.dim);
    printf("Layers: %d\n", config.n_layers);
    printf("Heads: %d\n", config.n_heads);
    printf("KV Heads: %d\n", config.n_kv_heads);
    printf("Vocab: %d\n", config.vocab_size);
    printf("Seq Len: %d\n", config.seq_len);
    
    printf("\n--- Running Inference ---\n");
    
    int token = 1;
    int pos = 0;
    
    float* content_row = weights.token_embedding_table + token * config.dim;
    memcpy(state.x, content_row, config.dim * sizeof(float));
    
    // Forward pass
    for (int layer = 0; layer < config.n_layers; layer++) {
        transformer_block(state.x, &state, &weights, &config, layer, pos);
    }
    
    // Final normalization
    RMSNorm(state.x, state.x, weights.rms_final_weight, config.dim);
    
    // Classifier (get logits)
    naive_matmul(state.logits, state.x, weights.w_cls, config.vocab_size, config.dim);
    
    // Find the token with highest probability (argmax)
    int next_token = 0;
    float max_val = state.logits[0];
    for (int i = 1; i < config.vocab_size; i++) {
        if (state.logits[i] > max_val) {
            max_val = state.logits[i];
            next_token = i;
        }
    }
    
    printf("Input Token: %d\n", token);
    printf("Next Token: %d (logit: %f)\n", next_token, max_val);
    printf("First 5 logits: ");
    for (int i = 0; i < 5; i++) {
        printf("%f ", state.logits[i]);
    }
    printf("\n");


    free_model_file(data, file_size);
    free_run_state(&state);
    printf("\nModel Unloaded. Exiting.\n");

    return 0;
}