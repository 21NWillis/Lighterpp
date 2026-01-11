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
    if (argc < 3) {
        printf("Usage: %s <model_path> <tokenizer_path> [temperature]\n", argv[0]);
        return 1;
    }
    
    float temperature = (argc >= 4) ? (float)atof(argv[3]) : 0.9f;
    float topp = 0.9f;
    srand(time(NULL));

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

    printf("Loading tokenizer: %s\n", argv[2]);
    
    Tokenizer tokenizer;
    load_tokenizer(&tokenizer, argv[2], config.vocab_size);

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
    int steps = 256;
    clock_t start = clock();

    for (int pos = 0; pos < steps; pos++) {
        token = forward(token, pos, &state, &weights, &config, temperature, topp);
        char* token_str = decode_token(&tokenizer, token);
        printf("%s", token_str);
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
    free_tokenizer(&tokenizer);
    free_model_file(data, file_size);
    free_run_state(&state);
    printf("\nModel Unloaded. Exiting.\n");

    return 0;
}