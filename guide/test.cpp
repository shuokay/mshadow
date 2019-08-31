/**
 * @file test.cpp
 * @author Yushu Gao
 * @brief test
 *
 * @copyright Copyright (c) 2019
 *
 */
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
int main(void) {
  test_shift();
  return 0;
}