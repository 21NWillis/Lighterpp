#include "tensor.h"

Tensor* allocate_tensor(int dim0, int dim1, int dim2, int dim3) {
    Tensor* t = (Tensor*)malloc(sizeof(Tensor));

    //Batch, Heads, Sequence, Dimensions
    t->shape[0] = dim0;
    t->shape[1] = dim1;
    t->shape[2] = dim2;
    t->shape[3] = dim3;

    t->size = dim0 * dim1 * dim2 * dim3;
    t->data = (float*)calloc(t->size, sizeof(float));

    if (t->data == NULL) {
        printf("Error: Memory allocation failed\n");
        exit(1);
    }

    return t;
}

void free_tensor(Tensor* t) {
    if (t->data != NULL) {
        free(t->data);
        free(t);
    }
}

void print_tensor(Tensor* t, int n_items, const char* label) {
    printf("%s: ", label);
    for (int i = 0; i < 20 && i < n_items; i++) {
        printf("%f ", t->data[i]);
    }
    printf("\n");
}
