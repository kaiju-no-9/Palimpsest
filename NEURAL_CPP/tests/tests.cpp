#include "nn/tensor.h"
#include <iostream>
#include <cassert>
#include <memory>

int main() {
    std::cout << "--- Starting Tensor Tests ---\n";

    Tensor t0(42.0f);
    assert(t0.shape().empty());
    assert(t0.item() == 42.0f);
    std::cout << "Scalar: " << t0 << "\n";

    Tensor t1(std::vector<float>{1.0f, 2.0f, 3.0f});
    assert(t1.shape().size() == 1);
    assert(t1.shape()[0] == 3);
    assert(t1(0) == 1.0f);
    assert(t1(2) == 3.0f);
    std::cout << "1D Vector: " << t1 << "\n";

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

    auto s1 = std::make_shared<Tensor>(5.0f);
    auto s2 = std::make_shared<Tensor>(3.0f);
    auto res_ss = (*s1) + s2;
    assert(res_ss->shape().empty());
    assert(res_ss->item() == 8.0f);
    std::cout << "Scalar + Scalar (5 + 3): " << *res_ss << "\n";

    auto v1 = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f, 3.0f});
    auto res_sv = (*s1) + v1;
    assert(res_sv->shape().size() == 1);
    assert((*res_sv)(0) == 6.0f);
    assert((*res_sv)(1) == 7.0f);
    assert((*res_sv)(2) == 8.0f);
    std::cout << "Scalar + 1D Vector (5 + [1, 2, 3]): " << *res_sv << "\n";

    auto m1 = std::make_shared<Tensor>(std::vector<std::vector<float>>{
        {1.0f, 2.0f},
        {3.0f, 4.0f}
    });
    auto res_sm = (*s1) + m1;
    assert(res_sm->shape().size() == 2);
    assert((*res_sm)(0, 0) == 6.0f);
    assert((*res_sm)(1, 1) == 9.0f);
    std::cout << "Scalar + 2D Matrix (5 + [[1,2],[3,4]]):\n" << *res_sm << "\n";

    auto res_vs = (*v1) + s1;
    assert(res_vs->shape().size() == 1);
    assert((*res_vs)(0) == 6.0f);
    assert((*res_vs)(2) == 8.0f);
    std::cout << "1D Vector + Scalar ([1, 2, 3] + 5): " << *res_vs << "\n";

    auto v2 = std::make_shared<Tensor>(std::vector<float>{10.0f, 20.0f, 30.0f});
    auto res_vv = (*v1) + v2;
    assert(res_vv->shape().size() == 1);
    assert((*res_vv)(0) == 11.0f);
    assert((*res_vv)(1) == 22.0f);
    assert((*res_vv)(2) == 33.0f);
    std::cout << "1D Vector + 1D Vector ([1,2,3] + [10,20,30]): " << *res_vv << "\n";

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

    try {
        auto v_mismatch = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f});
        auto fail_res = (*v1) + v_mismatch;
        assert(false && "Should have thrown invalid_argument for dimension mismatch");
    } catch (const std::invalid_argument &e) {
        std::cout << "Caught expected error on mismatched dimension: " << e.what() << "\n";
    }

    std::cout << "\n--- Testing Tensor::operator* (Shared Pointers) ---\n";

 
    auto mult_ss = (*s1) * s2;
    assert(mult_ss->shape().empty());
    assert(mult_ss->item() == 15.0f);
    std::cout << "1. Scalar * Scalar (5 * 3): " << *mult_ss << "\n";


    auto mult_sv = (*s1) * v1;
    assert(mult_sv->shape().size() == 1);
    assert(mult_sv->shape()[0] == 3);
    assert((*mult_sv)(0) == 5.0f);
    assert((*mult_sv)(1) == 10.0f);
    assert((*mult_sv)(2) == 15.0f);
    std::cout << "2. Scalar * 1D Vector (5 * [1, 2, 3]): " << *mult_sv << "\n";

   
    auto mult_sm = (*s1) * m1;
    assert(mult_sm->shape().size() == 2);
    assert(mult_sm->shape()[0] == 2 && mult_sm->shape()[1] == 2);
    assert((*mult_sm)(0, 0) == 5.0f);
    assert((*mult_sm)(0, 1) == 10.0f);
    assert((*mult_sm)(1, 0) == 15.0f);
    assert((*mult_sm)(1, 1) == 20.0f);
    std::cout << "3. Scalar * 2D Matrix (5 * [[1,2],[3,4]]):\n" << *mult_sm << "\n";


    auto mult_vs = (*v1) * s1;
    assert(mult_vs->shape().size() == 1);
    assert(mult_vs->shape()[0] == 3);
    assert((*mult_vs)(0) == 5.0f);
    assert((*mult_vs)(1) == 10.0f);
    assert((*mult_vs)(2) == 15.0f);
    std::cout << "4. 1D Vector * Scalar ([1, 2, 3] * 5): " << *mult_vs << "\n";

    auto mult_vv = (*v1) * v2;
    assert(mult_vv->shape().size() == 1);
    assert(mult_vv->shape()[0] == 3);
    assert((*mult_vv)(0) == 10.0f);
    assert((*mult_vv)(1) == 40.0f);
    assert((*mult_vv)(2) == 90.0f);
    std::cout << "5. 1D Vector * 1D Vector ([1,2,3] * [10,20,30]): " << *mult_vv << "\n";

 
    try {
        auto v_mismatch = std::make_shared<Tensor>(std::vector<float>{1.0f, 2.0f});
        auto fail_res = (*v1) * v_mismatch;
        assert(false && "Should have thrown invalid_argument for 1D dimension mismatch");
    } catch (const std::invalid_argument &e) {
        std::cout << "6. Caught expected 1D mismatch error: " << e.what() << "\n";
    }

 
    auto mult_mm = (*m1) * m2;
    assert(mult_mm->shape().size() == 2);
    assert(mult_mm->shape()[0] == 2 && mult_mm->shape()[1] == 2);
    assert((*mult_mm)(0, 0) == 10.0f);
    assert((*mult_mm)(0, 1) == 40.0f);
    assert((*mult_mm)(1, 0) == 90.0f);
    assert((*mult_mm)(1, 1) == 160.0f);
    std::cout << "7. 2D Matrix * 2D Matrix:\n" << *mult_mm << "\n";

  
    try {
        auto m_mismatch = std::make_shared<Tensor>(std::vector<std::vector<float>>{
            {1.0f, 2.0f, 3.0f},
            {4.0f, 5.0f, 6.0f}
        });
        auto fail_res = (*m1) * m_mismatch;
        assert(false && "Should have thrown invalid_argument for 2D dimension mismatch");
    } catch (const std::invalid_argument &e) {
        std::cout << "8. Caught expected 2D mismatch error: " << e.what() << "\n";
    }

    
    try {
        auto fail_res = (*v1) * m1;
        assert(false && "Should have thrown invalid_argument for unsupported rank combination");
    } catch (const std::invalid_argument &e) {
        std::cout << "9. Caught expected rank combination error (1D * 2D): " << e.what() << "\n";
    }

    std::cout << "\nAll tests passed successfully!\n";
    return 0;
}


