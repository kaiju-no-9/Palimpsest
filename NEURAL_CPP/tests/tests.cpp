#include "nn/tensor.h"
#include <iostream>
#include <cassert>
#include <memory>

int main() {
    std::cout << "--- Starting Tensor Tests ---\n";

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

    std::cout << "Testing Tensor::operator+ (Shared Pointers) ";

    // 1. Scalar + Scalar
    auto s1 = std::make_shared<Tensor>(5.0f);
    auto s2 = std::make_shared<Tensor>(3.0f);
    auto res_ss = (*s1) + s2;
    assert(res_ss->shape().empty());
    assert(res_ss->item() == 8.0f);
    std::cout << "Scalar + Scalar (5 + 3): " << *res_ss << "\n";

    // 2. Scalar + 1D Vector
    auto v1 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f});
    auto res_sv = (*s1) + v1;
    assert(res_sv->shape().size() == 1);
    assert((*res_sv)(0) == 6.0f);
    assert((*res_sv)(1) == 7.0f);
    assert((*res_sv)(2) == 8.0f);
    std::cout << "Scalar + 1D Vector (5 + [1, 2, 3]): " << *res_sv << "\n";

    // 3. Scalar + 2D Matrix
    auto m1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    });
    auto res_sm = (*s1) + m1;
    assert(res_sm->shape().size() == 2);
    assert((*res_sm)(0, 0) == 6.0f);
    assert((*res_sm)(1, 1) == 9.0f);
    std::cout << "Scalar + 2D Matrix (5 + [[1,2],[3,4]]):\n" << *res_sm << "\n";

    // 4. 1D Vector + Scalar
    auto res_vs = (*v1) + s1;
    assert(res_vs->shape().size() == 1);
    assert((*res_vs)(0) == 6.0f);
    assert((*res_vs)(2) == 8.0f);
    std::cout << "1D Vector + Scalar ([1, 2, 3] + 5): " << *res_vs << "\n";

    // 5. 1D Vector + 1D Vector
    auto v2 = std::make_shared<Tensor>(std::vector<float>{10.0f, 20.0f, 30.0f});
    auto res_vv = (*v1) + v2;
    assert(res_vv->shape().size() == 1);
    assert((*res_vv)(0) == 11.0f);
    assert((*res_vv)(1) == 22.0f);
    assert((*res_vv)(2) == 33.0f);
    std::cout << "1D Vector + 1D Vector ([1,2,3] + [10,20,30]): " << *res_vv << "\n";

    // 6. 2D Matrix + 2D Matrix
    auto m2 = std::make_shared<Tensor>(std::vector<std::vector<float>>{
        {10.0f, 20.0f},
        {30.0f, 40.0f}
    });
    auto res_mm = (*m1) + m2;
    assert(res_mm->shape().size() == 2);
    assert((*res_mm)(0, 0) == 11.0f);
    assert((*res_mm)(0, 1) == 22.0f);
    assert((*res_mm)(1, 0) == 33.0f);
    assert((*res_mm)(1, 1) == 44.0f);
    std::cout << "2D Matrix + 2D Matrix:\n" << *res_mm << "\n";

    // 7. Dimension Mismatch Error Test
    try {
        auto v_mismatch = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f});
        auto fail_res = (*v1) + v_mismatch;
        assert(false && "Should have thrown invalid_argument for dimension mismatch");
    } catch (const std::invalid_argument &e) {
        std::cout << "Caught expected error on mismatched dimension: " << e.what() << "\n";
    }

    std::cout << "\nTesting Tensor::operator* (Shared Pointers)\n";
    auto res_mult_ss = (*s1) * s2; // 5.0 * 3.0 = 15.0
    assert(res_mult_ss->item() == 15.0f);
    std::cout << "Scalar * Scalar (5 * 3): " << *res_mult_ss << "\n";

    auto res_mult_vv = (*v1) * v2; // [1,2,3] * [10,20,30] = [10,40,90]
    assert((*res_mult_vv)(0) == 10.0f);
    assert((*res_mult_vv)(1) == 40.0f);
    assert((*res_mult_vv)(2) == 90.0f);
    std::cout << "1D Vector * 1D Vector ([1,2,3] * [10,20,30]): " << *res_mult_vv << "\n";

    std::cout << "\nAll tests passed successfully!\n";
    return 0;
}


