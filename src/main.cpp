#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "loader.h"
#include "tensor.h"

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

    printf("\n--- Model Statistics ---\n");
    printf("Dim: %d\n", config.dim);
    printf("Layers: %d\n", config.n_layers);
    printf("Heads: %d\n", config.n_heads);
    printf("Vocab: %d\n", config.vocab_size);

    printf("\n--- Weight Verification ---\n");
    printf("Token Emb [0]:  %f\n", weights.token_embedding_table[0]);
    printf("Attn WQ [0]:    %f\n", weights.wq[0]);
    printf("Attn WK [0]:    %f\n", weights.wk[0]);
    printf("Final RMS [0]:  %f\n", weights.rms_final_weight[0]);
    
    state.key_cache[cache_elements - 1] = 1234.5f;
    printf("Key Cache Last Element: %f\n", state.key_cache[cache_elements - 1]);

    
    free_model_file(data, file_size);
    free_run_state(&state);
    printf("\nModel Unloaded. Exiting.\n");

    return 0;
}