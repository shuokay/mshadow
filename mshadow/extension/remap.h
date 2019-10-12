/**
 * @file remap.h
 * @author Yushu Gao
 * @brief remap
 *
 * @copyright Copyright (c) 2019
 *
 */
#ifndef MSHADOW_EXTENSION_REMAP_H_
#define MSHADOW_EXTENSION_REMAP_H_
#include "../extension.h"
namespace mshadow {
namespace expr {
template <typename SrcExp, typename MapExp, typename DType, int srcdim>
struct RemapExp
    : public MakeTensorExp<RemapExp<SrcExp, MapExp, DType, srcdim>, SrcExp, srcdim, DType> {
  const SrcExp& src_;
  const MapExp& map_;
  const index_t dst_height_;
  const index_t src_height_;
  explicit RemapExp(const SrcExp& src, const MapExp& map) : src_(src), map_(map) {
    Shape<srcdim> src_shape = ShapeCheck<srcdim, SrcExp>::Check(src_);
    Shape<3> map_shape = ShapeCheck<3, MapExp>::Check(map_);
    CHECK_EQ(map_shape.size(0), 2) << "y, x";
    this->shape_ = src_shape;
    this->shape_[srcdim - 2] = map_shape[srcdim - 2];
    this->shape_[srcdim - 1] = map_shape[srcdim - 1];
    dst_height_ = map_shape[1];
    src_height_ = src_shape[srcdim - 2];
  }
};

template <typename SrcExp, typename MapExp, typename DType, int src_type, int map_type>
inline RemapExp<SrcExp, MapExp, DType, ExpInfo<SrcExp>::kDim> remap(
    const Exp<SrcExp, DType, src_type>& src, const Exp<MapExp, DType, map_type>& map) {
  return RemapExp<SrcExp, MapExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), map.self());
}

template <typename SrcExp, typename MapExp, typename DType, int srcdim>
struct Plan<RemapExp<SrcExp, MapExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const RemapExp<SrcExp, MapExp, DType, srcdim>& e)
      : src_(MakePlan(e.src_)), map_(MakePlan(e.map_)) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t dst_w = j;
    const index_t dst_h = i % dst_height_;
    const index_t c = i / dst_height_;
    const DType src_h = map_.Eval(0 * dst_height_ + dst_h, dst_w);
    const DType src_w = map_.Eval(1 * dst_height_ + dst_h, dst_w);
    const int32_t src_h_floor = static_cast<int32_t>(src_h);
    const int32_t src_w_floor = static_cast<int32_t>(src_w);
    const int32_t src_h_ceil = src_h_floor + 1;
    const int32_t src_w_ceil = src_w_floor + 1;
    const DType top_left_value = src_.Eval(c * src_height_ + src_h_floor, src_w_floor);
    const DType top_right_value = src_.Eval(c * src_height_ + src_h_floor, src_w_ceil);
    const DType bottom_left_value = src_.Eval(c * src_height_ + src_h_ceil, src_w_floor);
    const DType bottom_right_value = src_.Eval(c * src_height_ + src_h_ceil, src_w_ceil);
    const float dy = src_h - src_h_floor;
    const float dx = src_w - src_w_floor;
    float result = top_left_value * (1 - dy) * (1 - dx) + bottom_right_value * dy * dx +
                   top_right_value * (1 - dy) * dx + bottom_left_value * dy * (1 - dx);
    return static_cast<DType>(result);
  }

 private:
  Plan<SrcExp, DType> src_;
  Plan<MapExp, DType> map_;
  const index_t dst_height_;
  const index_t src_height_;
};
}  // namespace expr
}  // namespace mshadow
#endif