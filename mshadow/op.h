/**
 * @file op.h
 * @author yushu.gao@horizon.ai
 * @brief define additional mshadow common math ops
 * 
 * @copyright Copyright (c) 2019
 * 
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

struct clip {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType x, DType min, DType max) {
    if (x < min) {
      return min;
    } else if (x > max) {
      return max;
    } else {
      return x;
    }
  }
  // Need to make sure bound >= 0
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType x, DType bound) {
    if (x < -bound) {
      return -bound;
    } else if (x > bound) {
      return bound;
    } else {
      return x;
    }
  }
};

struct abs {
  template <typename DType>
  MSHADOW_XINLINE static DType Map(DType x) {
    return x < 0 ? -x : x;
  }
};
}  // namespace op
}  // namespace mshadow
#endif