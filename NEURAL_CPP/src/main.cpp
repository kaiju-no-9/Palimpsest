#include "nn/tensor.h"
#include <vector>
#include <iostream>

int main () {
    Tensor test_tensor({ {1,2,3}, {4,5,6}, {7,8,9} });
    std::cout << test_tensor << std::endl; 
    return 0;  
}
