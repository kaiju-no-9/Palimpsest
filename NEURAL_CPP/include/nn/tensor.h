#pragma once
#include <vector> 
#include <cstddef>
#include <iostream>

class Tensor
{
private:
    std::vector<float> data_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> stride_;

public:
    // Constructors
    Tensor(float data);
    Tensor(std::vector<float> data);
    Tensor(std::vector<std::vector<float>> data);
    Tensor(std::vector<std::vector<std::vector<float>>> data);

    // Shape and data getters
    const std::vector<std::size_t> &shape() const { return shape_; }
    const std::vector<std::size_t> &stride() const { return stride_; }
    const std::vector<float> &data() const { return data_; }

    // Value accessors
    const float &item() const; 
    float &item(); 

    // Accessors
    const float &operator()(std::size_t i) const; 
    float &operator()(std::size_t i);
    const float &operator()(std::size_t i, std::size_t j) const;
    float &operator()(std::size_t i, std::size_t j); 

    // Stream printing
    friend std::ostream &operator<<(std::ostream &os, const Tensor &obj); 
};