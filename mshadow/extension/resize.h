/*!
 * Copyright (c) 2020 by Contributors
 * \file: resize.h
 * \date: 2020-10-05
 * \author: Yushu Gao
 * \brief:
 */

#ifndef MSHADOW_EXTENSION_RESIZE_H_
#define MSHADOW_EXTENSION_RESIZE_H_
#include "../extension.h"
#include "../op.h"
namespace mshadow {
namespace expr {
template <typename SrcExp, typename DType, int srcdim>
struct ResizeExp : public MakeTensorExp<ResizeExp<SrcExp, DType, srcdim>,
                                        SrcExp, srcdim, DType> {
  const SrcExp& src_;
  float start_y_;
  float start_x_;
  float step_y_;
  float step_x_;
  index_t src_height_;
  index_t src_width_;
  index_t out_height_;
  explicit ResizeExp(const SrcExp& src, index_t out_height, index_t out_width)
      : src_(src), out_height_(out_height) {
    Shape<srcdim> src_shape = ShapeCheck<srcdim, SrcExp>::Check(src_);
    this->shape_ = src_shape;
    this->shape_[srcdim - 2] = out_height;
    this->shape_[srcdim - 1] = out_width;
    step_y_ = src_shape[srcdim - 2] / static_cast<float>(out_height);
    step_x_ = src_shape[srcdim - 1] / static_cast<float>(out_width);
    start_y_ = (src_shape[srcdim - 2] / static_cast<float>(out_height) - 1) / 2;
    start_x_ = (src_shape[srcdim - 1] / static_cast<float>(out_width) - 1) / 2;
    src_height_ = src_shape[srcdim - 2];
    src_width_ = src_shape[srcdim - 1];
  }
};

template <typename SrcExp, typename DType, int etype>
inline ResizeExp<SrcExp, DType, ExpInfo<SrcExp>::kDim> resize(
    const Exp<SrcExp, DType, etype>& src, index_t out_height,
    index_t out_width) {
  return ResizeExp<SrcExp, DType, ExpInfo<SrcExp>::kDim>(src.self(), out_height,
                                                         out_width);
}

MSHADOW_XINLINE static bool in_bound(int32_t x, index_t low, index_t high) {
  return x >= low && x <= high;
}
template <typename SrcExp, typename DType, int srcdim>
struct Plan<ResizeExp<SrcExp, DType, srcdim>, DType> {
 public:
  explicit Plan(const ResizeExp<SrcExp, DType, srcdim>& e)
      : src_(MakePlan(e.src_)),
        start_y_(e.start_y_),
        start_x_(e.start_x_),
        step_y_(e.step_y_),
        step_x_(e.step_x_),
        src_height_(e.src_height_),
        src_width_(e.src_width_),
        out_height_(e.out_height_) {}
  MSHADOW_XINLINE DType Eval(index_t i, index_t j) const {
    const index_t dst_w = j;
    const index_t dst_h = i % out_height_;
    const index_t c = i / out_height_;
    const float src_w = start_x_ + dst_w * step_x_;
    const float src_h = start_y_ + dst_h * step_y_;
    int32_t src_h_floor = static_cast<int32_t>(std::floor(src_h));
    int32_t src_w_floor = static_cast<int32_t>(std::floor(src_w));
    int32_t src_h_ceil = src_h_floor + 1;
    int32_t src_w_ceil = src_w_floor + 1;
    auto get_src_coord = [](int32_t x, int32_t max) {
      return mshadow::op::minimum::Map(mshadow::op::maximum::Map(x, 0), max);
    };

    src_h_floor = get_src_coord(src_h_floor, src_height_ - 1);
    src_w_floor = get_src_coord(src_w_floor, src_width_ - 1);
    src_h_ceil = get_src_coord(src_h_ceil, src_height_ - 1);
    src_w_ceil = get_src_coord(src_w_ceil, src_width_ - 1);

    DType top_left_value = 0, top_right_value = 0, bottom_left_value = 0,
          bottom_right_value = 0;

    if (in_bound(src_h_floor, 0, src_height_ - 1) &&
        in_bound(src_w_floor, 0, src_width_ - 1)) {
      top_left_value = src_.Eval(c * src_height_ + src_h_floor, src_w_floor);
    }
    if (in_bound(src_h_floor, 0, src_height_ - 1) &&
        in_bound(src_w_ceil, 0, src_width_ - 1)) {
      top_right_value = src_.Eval(c * src_height_ + src_h_floor, src_w_ceil);
    }
    if (in_bound(src_h_ceil, 0, src_height_ - 1) &&
        in_bound(src_w_floor, 0, src_width_ - 1)) {
      bottom_left_value = src_.Eval(c * src_height_ + src_h_ceil, src_w_floor);
    }
    if (in_bound(src_h_ceil, 0, src_height_ - 1) &&
        in_bound(src_w_ceil, 0, src_width_ - 1)) {
      bottom_right_value = src_.Eval(c * src_height_ + src_h_ceil, src_w_ceil);
    }
    const float dy = src_h - src_h_floor;
    const float dx = src_w - src_w_floor;
    float result =
        top_left_value * (1 - dy) * (1 - dx) + bottom_right_value * dy * dx +
        top_right_value * (1 - dy) * dx + bottom_left_value * dy * (1 - dx);
    return static_cast<DType>(result);
  }

 private:
  Plan<SrcExp, DType> src_;
  const float start_y_;
  const float start_x_;
  const float step_y_;
  const float step_x_;
  const index_t src_height_;
  const index_t src_width_;
  const index_t out_height_;
};
}  // namespace expr
}  // namespace mshadow
#endif