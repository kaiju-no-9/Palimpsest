#pragma once
#include <vector> 
#include <cstddef>
#include <iostream>
#include <memory>
#include <functional>
// thisv part allows to create shared pointers to itself within member functions 
class Tensor:public std :: enable_shared_from_this<Tensor>
{
private:
    std::vector<float> data_;
    std::vector<std::size_t> shape_;
    std::vector<std::size_t> stride_;
    std :: vector<float> grad_;
    // will be used to update gratent  of the parent as per grantent of the child 
    std :: function <void(const std :: vector <float> &)> grad_fn_;
    // for traversing the  computaltional graph in backword order 
    std :: vector<std::shared_ptr<Tensor>> parents_of_tensors_;
    bool requires_grad_;

public:
    // Constructors
    Tensor(float data , bool requires_grad = false, std :: function<void(const std :: vector<float> &)> grad_fn_ = nullptr);
    Tensor(std::vector<float> data , bool requires_grad = false, std :: function<void(const std :: vector<float> &)> grad_fn_ = nullptr);
    Tensor(std::vector<std::vector<float>> data , bool requires_grad = false, std :: function<void(const std :: vector<float> &)> grad_fn_ = nullptr);
    Tensor(std::vector<std::vector<std::vector<float>>> data , bool requires_grad = false, std :: function<void(const std :: vector<float> &)> grad_fn_ = nullptr);

    // Shape and data getters
    // for time being not return  shape_ , stride_ as it will leed to double copy of data of tensor , so just make them public for now 
    const std::vector<std::size_t> &shape() const ; 
    const std::vector<std::size_t> &stride() const; 
    const std::vector<float> &data() const { return data_; };
    const std::vector<float>& grad() const;
    std::size_t numel() const;
    void add_to_grad(const std::vector<float> &grad_update);
    void zero_grad();
   
    const bool& requires_grad() const;
    const bool &require_bool() const;

    // Value accessors
    const float &item() const; 
    float &item(); 

    // Accessors
    const float &operator()(std::size_t i) const; 
    float &operator()(std::size_t i);
    const float &operator()(std::size_t i, std::size_t j) const;
    float &operator()(std::size_t i, std::size_t j); 

    // Operations
    std::shared_ptr<Tensor> operator+(std::shared_ptr<Tensor> other);
    std::shared_ptr<Tensor> operator*(std::shared_ptr<Tensor> other);

    // Stream printing
    friend std::ostream &operator<<(std::ostream &os, const Tensor &obj); 
};