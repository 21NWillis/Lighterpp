#include <iostream>
#include <stdlib.h>
#include "tensor.h"
#include "model.h"

int main(int argc, char* argv[]) {
    std::cout << "Inference Engine Initialized" << std::endl;
    
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <model_path>" << std::endl;
        return 1;
    }

    const char* model_path = argv[1];

    FILE *file = fopen(model_path, "rb");
    if (!file) {
        std::cerr << "Failed to open model file: " << model_path << std::endl;
        return 1;
    }

    Config config;
    if (fread(&config, sizeof(Config), 1, file) != 1) {
        std::cerr << "Failed to read config from model file: " << model_path << std::endl;
        fclose(file);
        return 1;
    }

    printf("Model Config Loaded:\n");
    printf("  dim:        %d\n", config.dim);
    printf("  hidden_dim: %d\n", config.hidden_dim);
    printf("  n_layers:   %d\n", config.n_layers);
    printf("  n_heads:    %d\n", config.n_heads);
    printf("  n_kv_heads: %d\n", config.n_kv_heads);
    printf("  vocab_size: %d\n", config.vocab_size);
    printf("  seq_len:    %d\n", config.seq_len);

    fclose(file);

    return 0;
}

