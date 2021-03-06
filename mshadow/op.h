/*!
 * Copyright (c) 2020 by Contributors
 * \file: op.h
 * \date: 2020-10-05
 * \author: Yushu Gao
 * \brief: 
*/

#ifndef MSHADOW_OP_H_
#define MSHADOW_OP_H_
#include "base.h"

namespace mshadow {
namespace op {
struct gt {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a > b ? DType(1) : DType(0);
  }
};

struct ge {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a >= b ? DType(1) : DType(0);
  }
};

struct lt {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a < b ? DType(1) : DType(0);
  }
};

struct le {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a <= b ? DType(1) : DType(0);
  }
};

struct eq {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a == b ? DType(1) : DType(0);
  }
};

struct ne {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a != b ? DType(1) : DType(0);
  }
};

struct abs {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType x) {
    return x < 0 ? -x : x;
  }
};

struct right_shift {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType value, DType shift) {
    return value >> shift;
  }
};

struct round_shift {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType value, DType shift) {
    DType half = DType(1) << (shift - 1);
    return (value + half) >> shift;
  }
};

struct left_shift {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType value, DType shift) {
    return value << shift;
  }
};

struct maximum {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a > b ? a : b;
  }
};

struct minimum {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType a, DType b) {
    return a < b ? a : b;
  }
};

}  // namespace op
}  // namespace mshadow
#endif