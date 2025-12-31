#include <iostream>
#include "tensor.h"

int main() {
    std::cout << "Inference Engine Initialized" << std::endl;
    
    Tensor* t = allocate_tensor(1, 1, 4, 4);

    t->data[0] = 5.5f;
    print_tensor(t, 20, "Input Tensor");

    free_tensor(t);

    return 0;
}

