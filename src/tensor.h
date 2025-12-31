#ifndef TENSOR_H
#define TENSOR_H

#include <stdio.h>
#include <stdlib.h>


struct Tensor {
    float *data;
    int shape[4];
    int size;
};

Tensor* allocate_tensor(int dim0, int dim1, int dim2, int dim3);
void free_tensor(Tensor* t);
void print_tensor(Tensor* t, int n_items, const char* label);






#endif

