#define EIGEN_USE_THREADS

// #include "tensorflow/core/framework/op.h"
#include "probe.h"
#include "tensorflow/core/framework/op_kernel.h"
// #include "tensorflow/core/framework/shape_inference.h"
// #include "tensorflow/core/framework/common_shape_fns.h"
// #include <cuda_runtime.h>
// #include <iostream>


using namespace tensorflow;
using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;


// Define Probe interface.
REGISTER_OP("Probe")
  .Input("input: float32")
  .Input("weights: float32")
  .Input("dims: float32")
  .Input("steps: float32")
  .Output("output: float32")
  // .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
  //   shape_inference::ShapeHandle input_shape;
  //   shape_inference::ShapeHandle weights_shape;
  //   shape_inference::ShapeHandle dims_shape;
  //   shape_inference::ShapeHandle steps_shape;

  //   c->WithRank(c->input(0), 5, &input_shape); // [num_batches, num_points, 6]
  //   c->WithRank(c->input(1), 3, &weights_shape); // [num_filters, num_probes, 3]
  //   c->WithRank(c->input(2), 1, &dims_shape); // length 3 array
  //   c->WithRank(c->input(3), 1, &steps_shape); // length 3 array

  //   shape_inference::DimensionHandle num_batches = c->Dim(input_shape, 0);
  //   shape_inference::DimensionHandle num_filters = c->Dim(weights_shape, 0);
  //   shape_inference::DimensionHandle x_steps = c->Dim(steps_shape, 0);
  //   shape_inference::DimensionHandle y_steps = c->Dim(steps_shape, 1);
  //   shape_inference::DimensionHandle z_steps = c->Dim(steps_shape, 2);

  //   shape_inference::ShapeHandle output_shape = c->MakeShape({num_batches, num_filters, x_steps, y_steps, z_steps});
  //   c->set_output(0, output_shape);

  //   return Status::OK();
  // });
  ;

template <typename T>
struct ProbeFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d, const int* sizes, const T* input, 
            const T* weights, const T* dims, const T* steps, T* output) {
  }
};

// template <typename T>
// struct ProbeFunctor<GPUDevice, T> {
//   void operator()(const GPUDevice& d, const int* sizes, const T* input, 
//             const T* weights, const T* dims, const T* steps, T* output);
// };

// CPU specialization of actual computation.
template <typename Device, typename T>
class ProbeOp : public OpKernel {
public:
  explicit ProbeOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& weights = context->input(1);
    const Tensor& dims = context->input(2);
    const Tensor& steps = context->input(3);
    OP_REQUIRES(context, input_tensor.dims()==3, 
      errors::InvalidArgument("Probe expects (batches, points, 6) input shape"));
    OP_REQUIRES(context, weights.dims()==3, 
      errors::InvalidArgument("Probe expects (filters, probes, 3) weights shape"));
    OP_REQUIRES(context, dims.dims()==1, 
      errors::InvalidArgument("Probe expects dims to be of dimension 1"));
    OP_REQUIRES(context, steps.dims()==1, 
      errors::InvalidArgument("Probe expects steps to be of dimension 1"));

    // Create an output tensor with the correct output shape
    // [num_batches, num_filters, x_steps, y_steps, z_steps]
    Tensor* output_tensor = NULL;
    int b = input_tensor.shape().dim_size(0);
    int f = weights.shape().dim_size(0);
    int x = steps.shape().dim_size(0);
    int y = steps.shape().dim_size(1);
    int z = steps.shape().dim_size(2);
    int p = input_tensor.shape().dim_size(1);
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,f,x,y,z},
                                                     &output_tensor));
    const int sizes[3] = {b,f,p};
    ProbeFunctor<Device, T>()(
        context->eigen_device<Device>(),
        sizes,
        input_tensor.flat<T>().data(),
        weights.flat<T>().data(),
        dims.flat<T>().data(),
        steps.flat<T>().data(),
        output_tensor->flat<T>().data());
    // const float* inp = &(input_tensor.flat<float>()(0));
    // const float* out = &(output_tensor->flat<float>()(0));
    // probeLauncher(sizes, inp, weights.flat<float>().data(),
    //   dims.flat<float>().data(), steps.flat<float>().data(), out);
  }
};

// Register the CPU kernels.
#define REGISTER_CPU(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("Probe").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      ProbeOp<CPUDevice, T>);
REGISTER_CPU(float);
//REGISTER_CPU(int32);


#define REGISTER_GPU(T)                                                                 \
  REGISTER_KERNEL_BUILDER(Name("Probe").Device(DEVICE_GPU).TypeConstraint<T>("T"),      \
    ProbeOp<GPUDevice, T>);
REGISTER_GPU(float);
//REGISTER_GPU(int32);


