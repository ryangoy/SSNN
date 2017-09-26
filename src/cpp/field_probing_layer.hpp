#ifndef TENSORFLOW_FIELD_PROBING_LAYER_HPP_
#define TENSORFLOW_FIELD_PROBING_LAYER_HPP_


#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/platform/mem.h"
#include "tensorflow/core/util/tensor_format.h"

#if GOOGLE_CUDA
#include "tensorflow/core/kernels/conv_ops_gpu.h"
#include "tensorflow/core/platform/stream_executor.h"
#endif  // GOOGLE_CUDA

namespace tensorflow {

// Forward declaration.
class OpKernelContext;

template <typename Device, typename T>
struct LaunchProbe3DOp {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, int row_stride,
                  int col_stride, const Padding& padding, Tensor* output,
                  TensorFormat data_format);
}

#ifdef GOOGLE_CUDA
template <typename T>
struct LaunchProbe3DOp<Eigen::GpuDevice, T> {
  void operator()(OpKernelContext* ctx, bool use_cudnn, bool cudnn_use_autotune,
                  const Tensor& input, const Tensor& filter, int row_stride,
                  int col_stride, const Padding& padding, Tensor* output,
                  TensorFormat data_format);
};
#endif  // GOOGLE_CUDA

// Used to keep track of persistent memory buffers used within the op.
// It uses malloc and free to avoid the time cost of initializing the memory.
template <class T, size_t size>
struct Im2ColBufferResource : public ResourceBase {
  Im2ColBufferResource<T, size>() {
    data = static_cast<T*>(port::Malloc(size * sizeof(T)));
  }
  ~Im2ColBufferResource<T, size>() { port::Free(data); }
  // This mutex ensures that only a single operation at a time is able to use
  // the buffer memory held by this resource.
  mutex mu;
  T* data;
  string DebugString() { return "Im2ColBufferResource"; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONV_OPS_H