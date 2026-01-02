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

void checkpoint_init_weights(transformerWeights* w, Config* p, float* ptr) {
    int head_size = p->dim/ p->n_heads;

    w->token_embedding_table = ptr;
    ptr += p->vocab_size * p->dim;

    w->rms_att_weight = ptr;
    ptr += p->n_layers * p->dim;

    w->wq = ptr;
    ptr += p->n_layers * p->dim * (p->n_heads * head_size);

    w->wk = ptr;
    ptr += p->n_layers * p->dim * (p->n_heads * head_size);

    w->wv = ptr;
    ptr += p->n_layers * p->dim * (p->n_heads * head_size);

    w->wo = ptr;
    ptr += p->n_layers * (p->n_heads * head_size) * p->dim;

    w->rms_ffn_weight = ptr;
    ptr += p->n_layers * p->dim;

    w->w1 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;

    w->w2 = ptr;
    ptr += p->n_layers * p->hidden_dim * p->dim;

    w->w3 = ptr;
    ptr += p->n_layers * p->dim * p->hidden_dim;

    w->rms_final_weight = ptr;
    ptr += p->dim;

    int head_size_bytes = head_size * sizeof(float);
    ptr += p->seq_len * head_size;
    ptr += p->seq_len * head_size; 

    // w_cls (Classifier / Un-embedding)
    // In most small Llama models, the weights to turn the vector back into a word
    // are SHARED with the weights used to turn the word into a vector.
    // This is called "Weight Tying".
    // We point w_cls back to the start of the array.
    w->w_cls = w->token_embedding_table;
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

    float* weightspointer = (float*)((char*)addr + sizeof(Config));

    *file_size_out = file_size;

    close(file);

    std::cout << "Model loaded successfully." << std::endl;
    return weightspointer;
}

void free_model_file(float* data, size_t file_size) {
    void* mmap_start = (void*)((char*)data - sizeof(Config));
    munmap(mmap_start, file_size);
}