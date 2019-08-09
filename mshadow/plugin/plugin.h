#ifndef MSHADOW_PLUGIN_H_
#define MSHADOW_PLUGIN_H_
#include "../tensor.h"
#ifdef __CUDACC__
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#endif
namespace mshadow {
namespace plugin {
#ifdef __CUDACC__
namespace cuda {
/*! \brief suggested thread number(logscale) for mapping kernel */
const int kBaseThreadBits = 8;
/*! \brief suggested thread number for mapping kernel */
const int kBaseThreadNum = 1 << kBaseThreadBits;
/*! \brief maximum value of grid */
const int kMaxGridNum = 65535;
/*! \brief maximum value of grid within each dimension */
const int kMaxGridDim = 65535;
/*! \brief suggested grid number for mapping kernel */
const int kBaseGridNum = 1024;
}  // namespace cuda
#define MSHADOW_CUDA_POST_KERNEL_CHECK(x)                                                  \
  /* Code block avoids redefinition of cudaError_t err */                                  \
  do {                                                                                     \
    cudaError err = cudaPeekAtLastError();                                                 \
    CHECK_EQ(err, cudaSuccess) << "Name: " << #x << " ErrStr:" << cudaGetErrorString(err); \
  } while (0)

template <typename OP, typename... Args>
__global__ void mxnet_generic_kernel(int N, Args... args) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += blockDim.x * gridDim.x) {
    OP::Map(i, args...);
  }
}

template <typename OP>
struct Kernel<OP, mshadow::gpu> {
  /*! \brief Launch GPU kernel */
  template <typename... Args>
  inline static void Launch(mshadow::Stream<mshadow::gpu>* s, int N, Args... args) {
    if (0 == N) return;
    int ngrid = std::min(cuda::kMaxGridNum, (N + cuda::kBaseThreadNum - 1) / cuda::kBaseThreadNum);
    mxnet_generic_kernel<OP, Args...>
        <<<ngrid, cuda::kBaseThreadNum, 0, mshadow::Stream<mshadow::gpu>::GetStream(s)>>>(N,
                                                                                          args...);
    MSHADOW_CUDA_POST_KERNEL_CHECK(mxnet_generic_kernel);
  }
};
#endif

#define MSHADOW_FORCE_INLINE inline __attribute__((always_inline))
#ifdef __CUDACC__
#define MSHADOW_XINLINE MSHADOW_FORCE_INLINE __device__ __host__
#else
#define MSHADOW_XINLINE MSHADOW_FORCE_INLINE
#endif
template <typename OP, typename xpu>
struct Kernel;
template <typename OP>
struct Kernel<OP, mshadow::cpu> {
  template <typename... Args>
  inline static bool Launch(mshadow::Stream<mshadow::cpu>*, const size_t N, Args... args) {
    // #pragma omp parallel for num_threads(8)
    for (size_t i = 0; i < static_cast<size_t>(N); ++i) {
      OP::Map(i, args...);
    }
    return true;
  }
};

/* \brief Compute flattened index given coordinates and shape. */
template <int ndim>
MSHADOW_XINLINE size_t ravel(const mshadow::Shape<ndim>& coord, const mshadow::Shape<ndim>& shape) {
  size_t ret = 0;
#pragma unroll
  for (int i = 0; i < ndim; ++i) {
    ret = ret * shape[i] + (shape[i] > coord[i]) * coord[i];
  }
  return ret;
}

/* Compute coordinates from flattened index given shape */
template <int ndim>
MSHADOW_XINLINE mshadow::Shape<ndim> unravel(const size_t idx, const mshadow::Shape<ndim>& shape) {
  mshadow::Shape<ndim> ret;
#pragma unroll
  for (int i = ndim - 1, j = idx; i >= 0; --i) {
    auto tmp = j / shape[i];
    ret[i] = j - tmp * shape[i];
    j = tmp;
  }
  return ret;
}
}  // namespace plugin
}  // namespace mshadow

#endif
