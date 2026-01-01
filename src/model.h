#ifndef MODEL_H
#define MODEL_H

#include "tensor.h"

struct Config {

    int dim;
    int hidden_dim;
    int n_layers;
    int n_heads;
    int n_kv_heads;
    int vocab_size;
    int seq_len;
};

struct transformerWeights {
    //TODO Later
};

#endif