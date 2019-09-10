/**
 * @file map_if.h
 * @author Yushu Gao
 * @brief
 *
 * @copyright Copyright (c) 2019
 *
 */

#ifndef MSHADOW_EXTENSION_MAP_IF_H_
#define MSHADOW_EXTENSION_MAP_IF_H_
#include "../extension.h"
namespace mshadow {
namespace expr {
template <typename SrcExp, typename MaskExp, typename DefaultExp, typename DType, int srcdim>
struct MapIfExp : public MakeTensorExp<MapIfExp<SrcExp, MaskExp, DefaultExp, DType, srcdim>, SrcExp,
                                       srcdim, DType> {
  const SrcExp& src_;
  const MaskExp& mask_;
  const DefaultExp& default_;
  explicit MapIfExp(const SrcExp& src_, const MaskExp& mask_, const DefaultExp& default_)
      : src_(src_), mask_(mask_), default_(default_) {
    this->shape_ = ShapeCheck<srcdim, SrcExp>::Check(src_);
  }
};

template <typename SrcExp, typename MaskExp, typename DefaultExp, typename DType, int src_type,
          int mask_type, int default_type>
inline MapIfExp<SrcExp, MaskExp, DefaultExp, DType, ExpInfo<SrcExp>::kDim> map_if(
    const Exp<SrcExp, DType, src_type>& src, const Exp<MaskExp, DType, mask_type>& mask,
    const Exp<DefaultExp, DType, default_type>& default_value) {
  return MapIfExp<SrcExp, MaskExp, DefaultExp, DType, ExpInfo<SrcExp>::kDim>(
      src.self(), mask.self(), default_value.self());
}

template <typename SrcExp, typename MaskExp, typename DefaultExp, typename DType, int srcdim>
struct Plan<MapIfExp<SrcExp, MaskExp, DefaultExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const MapIfExp<SrcExp, MaskExp, DefaultExp, DType, srcdim>& e)
      : src_(MakePlan(e.src_)), mask_(MakePlan(e.mask_)), default_(MakePlan(e.default_)) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    return mask_.Eval(i, j) ? src_.Eval(i, j) : default_.Eval(i, j);
  }
  Plan<SrcExp, DType> src_;
  Plan<MaskExp, DType> mask_;
  Plan<DefaultExp, DType> default_;
};
}  // namespace expr
}  // namespace mshadow

#endif