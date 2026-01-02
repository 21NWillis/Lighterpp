

#include <stdio.h>
#include <stdlib.h>
#include "model.h"
#include "loader.h"
#include "tensor.h"
#include "ops.h"
#include <math.h>


int main(int argc, char** argv) {
    Config config;
    transformerWeights weights;
    size_t file_size = 0;

    if (argc < 2) {
        printf("Usage: %s <model_path>\n", argv[0]);
        return 1;
    }

    float* data = load_model_file(argv[1], &config, &file_size);
    if (!data) return 1;

    checkpoint_init_weights(&weights, &config, data);

    printf("\n--- Verification ---\n");
    printf("Config: dim=%d, layers=%d\n", config.dim, config.n_layers);
    printf("First weight (Embed): %f\n", weights.token_embedding_table[0]);
    printf("First weight (Attn):  %f\n", weights.wq[0]);
    
    printf("Final RMS weight:     %f\n", weights.rms_final_weight[config.dim - 1]);

    float *x = (float *)malloc(config.dim*sizeof(float));
    for (int i = 0; i < config.dim; i++) {
        x[i] = 1.0f;
    }

    float *out = (float*)malloc(config.vocab_size*sizeof(float));

    printf("Running matmul on real weights...\n");

    naive_matmul(out, x, weights.token_embedding_table, config.vocab_size, config.dim);

    printf("Logits[0]: %f\n", out[0]);
    printf("Logits[100]: %f\n", out[100]);
    printf("Logits[last]: %f\n", out[config.vocab_size - 1]);

    free(out);
    free(x);

    free_model_file(data, file_size);
    

    return 0;
}