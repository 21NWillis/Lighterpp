

#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "loader.h"
#include "tensor.h"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        printf("Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    Config config;
    transformerWeights weights;
    size_t file_size = 0; 

    float* data = load_model_file(argv[1], &config, &file_size);
    if (!data) return 1;

    checkpoint_init_weights(&weights, &config, data);

    printf("\n--- Verification ---\n");
    printf("Config: dim=%d, layers=%d\n", config.dim, config.n_layers);
    printf("First weight (Embed): %f\n", weights.token_embedding_table[0]);
    printf("First weight (Attn):  %f\n", weights.wq[0]);
    
    printf("Final RMS weight:     %f\n", weights.rms_final_weight[config.dim - 1]);

    free_model_file(data, file_size);
    
    return 0;
}