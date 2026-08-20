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
    // vectors ; 
    Tensor(float data);
    Tensor(std::vector<float> data);
    Tensor(std::vector<std::vector<float>> data);
    Tensor(std::vector<std::vector<std::vector<float>>> data);
    // input feild for getting the value of the tensor ; 
     const float &item() const ; 
     float &item() ; 
    // accessors ; 
     const float &operator()(std::size_t i) const ; 
     const float &operator() (std::size_t i , std::size_t j ) const ;
     const float &operator()(std :: size_t i , std::size_t j ) ; 
     // testing 
     friend std :: ostream & operator<< (std :: ostream &os , const Tensor &obj  ) ; 
};