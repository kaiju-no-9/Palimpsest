#include "nn/tensor.h"

#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

// Row-major serialization
// Scalar tensor
Tensor::Tensor(float data) : data_{data}, shape_{}, stride_{} {}

// 1D tensor (vector)
Tensor::Tensor(std::vector<float> data) : data_{data}, shape_{data.size()}, stride_{1} {}

// 2D tensor (matrix)
Tensor::Tensor(std::vector<std::vector<float>> data)
    : shape_{data.size(), data.empty() ? 0 : data[0].size()},
      stride_{data.empty() ? 0 : data[0].size(), 1} {
    if (data.empty()) return;
    std::size_t n_expected_column = data[0].size();
    for (std::size_t i = 0; i < data.size(); ++i) {
        if (data[i].size() != n_expected_column) {
            throw std::invalid_argument("rows are not matching");
        }
    }
    for (std::size_t i = 0; i < data.size(); ++i) {
        for (std::size_t j = 0; j < data[0].size(); ++j) {
            data_.push_back(data[i][j]);
        }
    }
}

// Tensor item accessors
const float &Tensor::item() const {
    if (data_.size() == 1) {
        return data_[0];
    }
    throw std::runtime_error("the method can only be used in case of single element tensors");
}

float &Tensor::item() {
    if (data_.size() == 1) {
        return data_[0];
    }
    throw std::runtime_error("the method can only be used in case of single element tensors");
}

// 1D Accessors
const float &Tensor::operator()(std::size_t i) const {
    if (shape_.size() == 0) {
        throw std::invalid_argument("cannot index into a scalar");
    }
    if (shape_.size() == 1) {
        if (i >= shape_[0]) {
            throw std::invalid_argument("index " + std::to_string(i) +
                                        " is out of bound for an array of size " +
                                        std::to_string(shape_[0]));
        }
        return data_[i];
    }
    throw std::invalid_argument("this is a 2D tensor, use 2D tensor accessors");
}

float &Tensor::operator()(std::size_t i) {
    return const_cast<float &>(static_cast<const Tensor &>(*this)(i));
}

// 2D Accessors
const float &Tensor::operator()(std::size_t i, std::size_t j) const {
    if (shape_.size() == 2) {
        if (i >= shape_[0]) {
            throw std::invalid_argument("row index " + std::to_string(i) +
                                        " is out of bound for an array of size " +
                                        std::to_string(shape_[0]));
        }
        if (j >= shape_[1]) {
            throw std::invalid_argument("col index " + std::to_string(j) +
                                        " is out of bound for an array of size " +
                                        std::to_string(shape_[1]));
        }
        return data_[i * stride_[0] + j * stride_[1]];
    }
    throw std::runtime_error(std::to_string(shape_.size()) + "D Tensor indexing not supported yet");
}

float &Tensor::operator()(std::size_t i, std::size_t j) {
    return const_cast<float &>(static_cast<const Tensor &>(*this)(i, j));
}
// method to view the shape andv stride fo the project :
// tesor.h lines : 22, 23
const std ::vector<std ::size_t> &Tensor::shape() const {
    return shape_;
}
const std ::vector<std ::size_t> &Tensor::stride() const {
    return stride_;
}
// Stream printing
std::ostream &operator<<(std::ostream &os, const Tensor &obj) {
    if (obj.shape().size() == 0) {
        return os << obj.item();
    }

    std::string string_rep = "[";
    if (obj.shape().size() == 1) {
        for (std::size_t i = 0; i < obj.shape()[0]; ++i) {
            string_rep += std::to_string(obj(i));
            if (i != obj.shape()[0] - 1) {
                string_rep += ", ";
            }
        }
        string_rep += "]";
    } else if (obj.shape().size() == 2) {
        for (std::size_t i = 0; i < obj.shape()[0]; ++i) {
            if (i > 0) string_rep += " ";
            string_rep += "[";
            for (std::size_t j = 0; j < obj.shape()[1]; ++j) {
                string_rep += std::to_string(obj(i, j));
                if (j != obj.shape()[1] - 1) {
                    string_rep += ", ";
                }
            }
            string_rep += "]";
            if (i != obj.shape()[0] - 1) {
                string_rep += ",\n";
            }
        }
        string_rep += "]";
    }
    return os << string_rep;
}

// shared_pointer for tensor op that we will perform:
std::shared_ptr<Tensor> Tensor::operator+(std::shared_ptr<Tensor> other) {
    // scaler + scaler
    if (shape_.size() == 0 && other->shape().size() == 0) {
        float result = item() + other->item();
        return std::make_shared<Tensor>(result);
    }
    // scaler + vector
    if (shape_.size() == 0 && other->shape().size() == 1) {
        std ::vector<float> data;
        for (std::size_t i = 0; i < other->shape()[0]; ++i) {
            data.push_back(item() + ((*other)(i)));
        }
        return std ::make_shared<Tensor>(data);
    }
    // scaler +  2d

    if (shape_.size() == 0 && other->shape().size() == 2) {
        std ::vector<std ::vector<float>> result;
        for (std ::size_t i = 0; i < other->shape()[0]; ++i) {
            std ::vector<float> row_data;
            for (std::size_t j = 0; j < other->shape()[1]; ++j) {
                row_data.push_back(item() + (*other)(i, j));
            }
            result.push_back(row_data);
        }
        return std ::make_shared<Tensor>(result);
    }

    //  1d  + scaler
    if (shape_.size() == 1 && other->shape().size() == 0) {
        std::vector<float> result;
        for (std::size_t i = 0; i < shape_[0]; ++i) {
            result.push_back((*this)(i) + other->item());
        }
        return std::make_shared<Tensor>(result);
    }
    // 1d+ 1d
    if (shape_[0] != other->shape()[0]) {
        throw std ::invalid_argument("the required dimension are invalid ");
    };
    if (shape_.size() == 1) {
        std ::vector<float> result;
        for (std ::size_t i = 0; i < shape_[0]; ++i) {
            result.push_back(operator()(i) + ((*other)(i)));
        }
        return std ::make_shared<Tensor>(result);
    }

    // 2d +2d
    else {
        if (shape_[1] != other->shape()[1]) {
            throw std ::invalid_argument("the required dimension are invalid ");
        }
        std ::vector<std ::vector<float>> result;
        for (std ::size_t i = 0; i < shape_[0]; ++i) {
            std ::vector<float> row_data;
            for (std ::size_t j = 0; j < shape_[1]; ++j) {
                row_data.push_back(operator()(i, j) + (*other)(i, j));
            }
            result.push_back(row_data);
        }
        return std ::make_shared<Tensor>(result);
    }
}

std::shared_ptr<Tensor> Tensor::operator*(std::shared_ptr<Tensor> other) {
    // scalar * scalar 
    if (shape_.size() == 0 && other->shape().size() == 0) {
        float result = item() * other->item();
        return std::make_shared<Tensor>(result);
    }
    // scalar * vector 
    if (shape_.size() == 0 && other->shape().size() == 1) {
        std::vector<float> data; 
        for (std::size_t i = 0; i < other->shape()[0]; ++i) {
            data.push_back(item() * ((*other)(i))); 
        }
        return std::make_shared<Tensor>(data); 
    }
    // scalar * 2d 
    if (shape_.size() == 0 && other->shape().size() == 2) {
        std::vector<std::vector<float>> result; 
        for (std::size_t i = 0; i < other->shape()[0]; ++i) {
            std::vector<float> row_data;
            for (std::size_t j = 0; j < other->shape()[1]; ++j) {
                row_data.push_back(item() * (*other)(i, j));
            }
            result.push_back(row_data);
        }
        return std::make_shared<Tensor>(result);
    }

    // 1d * scalar 
    if (shape_.size() == 1 && other->shape().size() == 0) {
        std::vector<float> result; 
        for (std::size_t i = 0; i < shape_[0]; ++i) {
            result.push_back((*this)(i) * other->item()); 
        } 
        return std::make_shared<Tensor>(result);
    } 
    // 1d * 1d 
    if (shape_.size() == 1 && other->shape().size() == 1) {
        if (shape_[0] != other->shape()[0]) {
            throw std::invalid_argument("the required dimension are invalid ");
        }
        std::vector<float> result; 
        for (std::size_t i = 0; i < shape_[0]; ++i) {
            result.push_back(operator()(i) * ((*other)(i)));
        }
        return std::make_shared<Tensor>(result);
    }
    // 2d * 2d 
    if (shape_.size() == 2 && other->shape().size() == 2) {
        if (shape_[0] != other->shape()[0] || shape_[1] != other->shape()[1]) {
            throw std::invalid_argument("the required dimension are invalid ");
        }
        std::vector<std::vector<float>> result; 
        for (std::size_t i = 0; i < shape_[0]; ++i) {
            std::vector<float> row_data; 
            for (std::size_t j = 0; j < shape_[1]; ++j) {
                row_data.push_back(operator()(i, j) * (*other)(i, j));
            } 
            result.push_back(row_data);
        } 
        return std::make_shared<Tensor>(result);
    }
    throw std::invalid_argument("the required dimension are invalid ");
}