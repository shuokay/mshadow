/**
 * @file clip.h
 * @author Yushu Gao
 * @brief clip the tensor to a certain range
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef MSHADOW_EXTENSION_CLIP_H_
#define MSHADOW_EXTENSION_CLIP_H_
#include "../extension.h"
namespace mshadow {
namespace expr {
/**
 * @brief clip expression, clip a tensor to a certain range
 *
 * @tparam SrcExp source expression to be clipped
 * @tparam MinExp min expression, the lower bound
 * @tparam MaxExp max expression, the upper bound
 * @tparam DType the type of elements
 * @tparam srcdim dimension of src
 */
template <typename SrcExp, typename MinExp, typename MaxExp, typename DType, int srcdim>
struct ClipExp
    : public MakeTensorExp<ClipExp<SrcExp, MinExp, MaxExp, DType, srcdim>, SrcExp, srcdim, DType> {
  /*! \brief operand */
  const SrcExp& src_;
  const MinExp& min_;
  const MaxExp& max_;
  /*! \brief constructor */
  explicit ClipExp(const SrcExp& src, const MinExp& min, const MaxExp& max)
      : src_(src), min_(min), max_(max) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
  }

  explicit ClipExp(const SrcExp& src, const DType min, const DType max)
      : src_(src), min_(ScalarExp<DType>(min)), max_(ScalarExp<DType>(max)) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
  }
};
/**
 * @brief clip expression, clip a tensor to a certain range
 *
 * @tparam SrcExp source expression to be clipped
 * @tparam MinExp min expression, the lower bound
 * @tparam MaxExp max expression, the upper bound
 * @tparam DType the type of elements
 * @tparam etype the expression type of src
 * @tparam mintype the expression type of min
 * @tparam maxtype the expression type of max
 * @param src the expression to be clipped
 * @param min the expression of lower bound
 * @param max the expression of upper bound
 * @return ClipExp<SrcExp, MinExp, MaxExp, DType, ExpInfo<SrcExp>::kDim>
 */
template <typename SrcExp, typename MinExp, typename MaxExp, typename DType, int etype, int mintype,
          int maxtype>
inline ClipExp<SrcExp, MinExp, MaxExp, DType, ExpInfo<SrcExp>::kDim> clip(
    const Exp<SrcExp, DType, etype>& src, const Exp<MinExp, DType, mintype>& min,
    const Exp<MaxExp, DType, maxtype>& max) {
  return ClipExp<SrcExp, MinExp, MaxExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), min.self(),
                                                                       max.self());
}
/**
 * @brief clip expression, clip a tensor to a certain range
 *
 * @tparam SrcExp source expression to be clipped
 * @tparam DType the type of elements
 * @tparam etype the expression type of src
 * @param src the expression to be clipped
 * @param min the lower bound
 * @param max the upper bound
 * @return ClipExp<SrcExp, ScalarExp<DType>, ScalarExp<DType>, DType, ExpInfo<SrcExp>::kDim>
 */
template <typename SrcExp, typename DType, int etype>
inline ClipExp<SrcExp, ScalarExp<DType>, ScalarExp<DType>, DType, ExpInfo<SrcExp>::kDim> clip(
    const Exp<SrcExp, DType, etype>& src, const DType min, const DType max) {
  return ClipExp<SrcExp, ScalarExp<DType>, ScalarExp<DType>, DType, ExpInfo<SrcExp>::kDim>(
      src.self(), min, max);
}
//----------------------
// Execution plan
//----------------------
template <typename SrcExp, typename MinExp, typename MaxExp, typename DType, int srcdim>
struct Plan<ClipExp<SrcExp, MinExp, MaxExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const ClipExp<SrcExp, MinExp, MaxExp, DType, srcdim>& e)
      : src_(MakePlan(e.src_)), min_(MakePlan(e.min_)), max_(MakePlan(e.max_)) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    DType src = src_.Eval(i, j);
    DType min = min_.Eval(i, j);
    DType max = max_.Eval(i, j);
    if (src < min) {
      return min;
    } else if (src > max) {
      return max;
    } else {
      return src;
    }
  }

 private:
  Plan<SrcExp, DType> src_;
  Plan<MaxExp, DType> min_;
  Plan<MinExp, DType> max_;
};
}  // namespace expr
}  // namespace mshadow
#endif  // MSHADOW_EXTENSION_CLIP_H_
