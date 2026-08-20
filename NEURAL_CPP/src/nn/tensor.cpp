#include "nn/tensor.h"
#include <stdexcept>
#include <vector>
// here i am imp row major seralization  just like used in pytorch
// scaler tensor 
Tensor :: Tensor(float data) : data_ {data}, shape_{} , stride_{} {}
//1d tensor ( vector )
Tensor :: Tensor(std::vector<float> data) : data_ {}, shape_{data.size()} , stride_{1} {}
// 2d tensor ( matrix )
Tensor :: Tensor(std::vector<std::vector<float>> data) 
      : shape_{data.size(), data[0].size()} , stride_{data[0].size(), 1} 
      {
        // dimentional matches 
        std:: size_t n_expected_column = data[0].size();
        for ( std :: size_t i=0;i<data.size();++i)
        {
          if ( data[i].size() != n_expected_column )
          {
            throw std::invalid_argument("rows are not matching");
          }
        }
        // storing data .. 
        for (std :: size_t i=0;i<data.size();++i)
        {
          for (std :: size_t j=0;j<data[0].size();++j)
          {
            data_.push_back(data[i][j]);
          }
        }
      }

     // tensor.item(): impo the function thst returns the value asa standerd no element ; 

const float &Tensor::item() const 
{
    if(data_.size()==1){
        return data_[0]; 
    }
    else {
        std :: runtime_error(" the method can only be used in case of single elementy only"); 
    }
}

const float &Tensor::operator()(std::size_t i ) const
{
  // here the dim is 1 for the first function 
  if(shape_.size() ==0  ){
    throw std::invalid_argument(" cannot intdex int a scaler  ") ;
  }
  if(shape_.size() == 1){
    throw std :: invalid_argument("index"  + std ::to_string(i)+ "is out of bound for an array of size "+std :: to_string(shape_[0])  ) ;
  }
  return data_[0]; 
   throw std :: invalid_argument(" this is a 2d sensor use a 2d  tensor accessors");

}

const float &Tensor::operator()(std::size_t i ,std::size_t j) const {
  // here the dim is 2 for the second function 
   if ( shape_.size() ==0){
    if ( i>= shape_[0]){
        throw std :: invalid_argument("row Index "+ std ::to_string(i) + "is out of bound for an array of size "+ std :: to_string(shape_[0])  ) ; 
    }
    if(j>= shape_[1]){
        throw std :: invalid_argument("col Index "+ std ::to_string(j) + "is out of bound for an array of size "+ std :: to_string(shape_[1])  ) ; 
    }
    return data_[i * stride_[0] + j * stride_[1] ];
   }
   throw std :: runtime_error(std::to_string(shape_.size()) + "d Tensor" + " indexing not supported yet"  ) ; 
   
}

//cutum testing endpoint ::
 std::ostream & operator<< (std :: ostream &os , const Tensor &obj ) {
    
 }