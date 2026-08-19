#include "nn/tensor.h"
#include <stdexcept>
#include <vector>

Tensor :: Tensor(float data) : data_ {data}, shape_{} , stride_{} {}
Tensor :: Tensor(std::vector<float> data) : data_ {}, shape_{} , stride_{1} {}
Tensor :: Tensor(std::vector<std::vector<float>> data) : data_ {}, shape_{data.size(), data[0].size()} , stride_{data[0].size(),1} {}

Tensor::Tensor(std::vector<std::vector<std::vector<float>>> data) {
   
}
