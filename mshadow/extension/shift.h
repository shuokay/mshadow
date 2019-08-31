/**
 * @file shift.h
 * @author Yushu Gao
 * @brief right shift, support floor and round approximation
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef MSHADOW_EXTENSION_SHIFT_H_
#define MSHADOW_EXTENSION_SHIFT_H_
#include "../extension.h"
namespace mshadow {
namespace expr {
namespace approx {
enum { kFloor, kRound };
}
template <typename SrcExp, typename ShiftValueExp, int dim, typename DType, int approx>
struct ShiftExp : public MakeTensorExp<ShiftExp<SrcExp, ShiftValueExp, dim, DType, approx>,
                                       SrcExp, dim, DType> {
  const SrcExp& src_;
  const ShiftValueExp& shift_value_;
  const int approx_;
  explicit ShiftExp(const SrcExp& src, const ShiftValueExp& shift_v, const DType shift_value)
      : src_(src), shift_value_(shift_value), approx_(approx) {
    this->shape_ = ShapeCheck<dim, SrcExp>::Check(src_);
  }
  explicit ShiftExp(const SrcExp& src, const DType shift_value)
      : src_(src), shift_value_(ScalarExp<DType>(shift_value)), approx_(approx) {
    this->shape_ = ShapeCheck<dim, SrcExp>::Check(src_);
  }
};
/**
 * @brief right shift
 *
 * @tparam approx kFloor or kRound
 * @tparam SrcExp
 * @tparam ShiftValueExp
 * @tparam DType
 * @tparam etype
 * @tparam stype
 * @param src
 * @param shift_value
 * @return ShiftExp<SrcExp, ShiftValueExp, ExpInfo<SrcExp>::kDim, DType, approx>
 */
template <int approx, typename SrcExp, typename ShiftValueExp, typename DType, int etype,
          int stype>
inline ShiftExp<SrcExp, ShiftValueExp, ExpInfo<SrcExp>::kDim, DType, approx> shift(
    const Exp<SrcExp, DType, etype>& src, const Exp<ShiftValueExp, DType, stype>& shift_value) {
  return ShiftExp<SrcExp, ShiftValueExp, ExpInfo<SrcExp>::kDim, DType, approx>(
      src.self(), shift_value.self());
}
/**
 * @brief right shift
 *
 * @tparam approx kFloor or kRound
 * @tparam SrcExp
 * @tparam DType
 * @tparam etype
 * @param src
 * @param shift_value
 * @return ShiftExp<SrcExp, ScalarExp<DType>, ExpInfo<SrcExp>::kDim, DType, approx>
 */
template <int approx, typename SrcExp, typename DType, int etype>
inline ShiftExp<SrcExp, ScalarExp<DType>, ExpInfo<SrcExp>::kDim, DType, approx> shift(
    const Exp<SrcExp, DType, etype>& src, const DType shift_value) {
  return ShiftExp<SrcExp, ScalarExp<DType>, ExpInfo<SrcExp>::kDim, DType, approx>(src.self(),
                                                                                       shift_value);
}

template <int approx, typename DType>
struct shift_impl;

template <typename DType>
struct shift_impl<approx::kFloor, DType> {
  MSHADOW_XINLINE static DType Map(DType data, DType shift_value) { return data >> shift_value; }
};

template <typename DType>
struct shift_impl<approx::kRound, DType> {
  MSHADOW_XINLINE static DType Map(DType data, DType shift_value) {
    DType half = DType(1) << (shift_value - 1);
    return (data + half) >> (shift_value);
  }
};
template <typename SrcExp, typename ShiftValueExp, int dim, typename DType, int approx>
struct Plan<ShiftExp<SrcExp, ShiftValueExp, dim, DType, approx>, DType> {
 public:
  explicit Plan(const ShiftExp<SrcExp, ShiftValueExp, dim, DType, approx>& e)
      : src_(MakePlan(e.src_)), shift_value_(MakePlan(e.shift_value_)) {}
  MSHADOW_XINLINE DType Eval(index_t y, index_t x) const {
    DType v = src_.Eval(y, x);
    DType s = shift_value_.Eval(y, x);
    return shift_impl<approx, DType>::Map(v, s);
  }

 private:
  Plan<SrcExp, DType> src_;
  Plan<ShiftValueExp, DType> shift_value_;
};
}  // namespace expr
}  // namespace mshadow
#endif