#pragma once
#include <vector> 
#include <cstddef>

class Tensor
{
private:
    std::vector<float> data_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> stride_;

public:
    Tensor(float data);
    Tensor(std::vector<float> data);
    Tensor(std::vector<std::vector<float>> data);
    Tensor(std::vector<std::vector<std::vector<float>>> data);
};