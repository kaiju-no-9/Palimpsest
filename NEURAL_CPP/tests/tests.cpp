#include "nn/tensor.h"
#include <iostream>
#include <cassert>

int main() {
    // 0D Scalar
    Tensor t0(42.0f);
    assert(t0.shape().empty());
    assert(t0.item() == 42.0f);
    std::cout << "Scalar: " << t0 << "\n";

    // 1D Vector
    Tensor t1(std::vector<float>{1.0f, 2.0f, 3.0f});
    assert(t1.shape().size() == 1);
    assert(t1.shape()[0] == 3);
    assert(t1(0) == 1.0f);
    assert(t1(2) == 3.0f);
    std::cout << "1D Vector: " << t1 << "\n";

    // 2D Matrix
    Tensor t2(std::vector<std::vector<float>>{
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    });
    assert(t2.shape().size() == 2);
    assert(t2.shape()[0] == 2 && t2.shape()[1] == 2);
    assert(t2(0, 0) == 1.0f);
    assert(t2(1, 1) == 4.0f);
    std::cout << "2D Matrix:\n" << t2 << "\n";

    std::cout << "\nAll tests passed successfully!\n";
    return 0;
}
