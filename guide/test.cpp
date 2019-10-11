/**
 * @file test.cpp
 * @author Yushu Gao
 * @brief test
 *
 * @copyright Copyright (c) 2019
 *
 */
#include <iostream>
#include "mshadow/tensor.h"
using namespace mshadow;
using namespace mshadow::expr;
template <typename xpu>
Stream<xpu>* GetSream() {
  static Stream<xpu>* s = NewStream<xpu>(0);
  return s;
}
void test_shift() {
  using xpu = cpu;
  Random<xpu, float> rand{123};
  Stream<xpu>* s = GetSream<xpu>();
  Shape<4> shape = Shape4(1, 2, 3, 4);
  Tensor<xpu, 4, int32_t> test = NewTensor<xpu, int32_t, 4>(shape, 0, true, s);
  test = tcast<int32_t>(rand.uniform(shape) * 1000);
  LOG(INFO) << test[0][0][0][0];
  test = shift<approx::kRound>(test, int32_t(3));
  LOG(INFO) << test[0][0][0][0];
}
template <typename DType>
void Print2DTensor(const mshadow::Tensor<cpu, 2, DType>& src) {
  for (int i = 0; i < src.size(0); ++i) {
    for (int j = 0; j < src.size(1); ++j) {
      std::cout << src[i][j] << "\t";
    }
    std::cout << std::endl;
  }
}

void test_resize() {
  using xpu = cpu;
  Random<xpu, float> rand{123};
  Stream<xpu>* s = GetSream<xpu>();
  float data[] = {1, 2,  4,  8,  16, 32, 64, 32, 5,  6,  8,  12, 20, 36, 68, 36,
                    9, 10, 12, 16, 24, 40, 72, 40, 13, 14, 16, 20, 28, 44, 76, 44};
  Shape<2> shape = Shape2(4, 8);
  Shape<2> oshape = Shape2(3, 3);
  Tensor<xpu, 2, float> src{data, shape};
  Tensor<xpu, 2, float> dst = NewTensor<xpu, float, 2>(oshape, float(0), true, s);
  dst = resize(src, 3, 3, resize_pad::kEdge, float(-1));
  Print2DTensor(src);
  Print2DTensor(dst);
}
int main(void) {
  // test_shift();
  test_resize();
  return 0;
}