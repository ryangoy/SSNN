// Functors for 3d probing.

#ifndef TENSORFLOW_KERNELS_PROBE_3D_H_
#define TENSORFLOW_KERNELS_PROBE_3D_H_

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/eigen_cuboid_convolution.h"

namespace tensorflow {
namespace functor {

// Applies a 3D convolution to a batch of multi-channel volumes.
template <typename Device, typename T>
struct CuboidConvolution;

typedef Eigen::ThreadPoolDevice CPUDevice;

template <typename T>
struct CuboidConvolution<CPUDevice, T> {
  void operator()(const CPUDevice& d, typename TTypes<T, 5>::Tensor output,
                  typename TTypes<T, 5>::ConstTensor input,
                  typename TTypes<T, 5>::ConstTensor filter, int stride_planes,
                  int stride_rows, int stride_cols,
                  const Eigen::PaddingType& padding) {
    output.device(d) = Eigen::CuboidConvolution(
        input, filter, stride_planes, stride_rows, stride_cols, padding);
  }
};

}  // namespace functor
}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONV_3D_H_
