#ifndef LOADER_H
#define LOADER_H

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "model.h"

float* load_model_file(const char* checkpoint_path, Config* config, size_t* file_size_out);
void free_model_file(float* data, size_t file_size);
void checkpoint_init_weights(transformerWeights* w, Config* p, float* ptr);

#ifdef USE_CUDA
void free_weights_cuda(transformerWeights* w);
#endif

#endif