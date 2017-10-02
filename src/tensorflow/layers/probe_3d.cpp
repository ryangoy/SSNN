#define EIGEN_USE_THREADS

#include "probe_3d.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/common_shape_fns.h"
#include <cuda_runtime.h>

using namespace tensorflow;

// Define Probe3D interface.
REGISTER_OP("Probe3D")
  .Input("input: float")
  .Input("weights: float")
  .Input("dims: float")
  .Input("steps: float")
  .Output("output: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle input_shape;
    shape_inference::ShapeHandle weights_shape;
    shape_inference::ShapeHandle dims_shape;
    shape_inference::ShapeHandle steps_shape;

    c->WithRank(c->input(0), 5, &input_shape); // [num_batches, num_points, 6]
    c->WithRank(c->input(1), 3, &weights_shape); // [num_filters, num_probes, 3]
    c->WithRank(c->input(2), 1, &dims_shape); // length 3 array
    c->WithRank(c->input(3), 1, &steps_shape); // length 3 array

    shape_inference::DimensionHandle num_batches = c->Dim(input_shape, 0);
    shape_inference::DimensionHandle num_filters = c->Dim(weights_shape, 0);
    shape_inference::DimensionHandle x_steps = c->Dim(steps_shape, 0);
    shape_inference::DimensionHandle y_steps = c->Dim(steps_shape, 1);
    shape_inference::DimensionHandle z_steps = c->Dim(steps_shape, 2);

    shape_inference::ShapeHandle output_shape = c->MakeShape({num_batches, num_filters, x_steps, y_steps, z_steps});
    c->set_output(0, output_shape);

    return Status::OK();
  });

template <typename Device, typename T>
class Probe3DGpuOp : public OpKernel {
public:
  explicit Probe3DGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {

    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& weights = context->input(1);
    const Tensor& dims = context->input(2);
    const Tensor& steps = context->input(3);

    OP_REQUIRES(context, input_tensor.dims()==3, 
      errors::InvalidArgument("Probe3D expects (batches, points, 6) input shape"));
    OP_REQUIRES(context, weights.dims()==3, 
      errors::InvalidArgument("Probe3D expects (filters, probes, 3) weights shape"));
    OP_REQUIRES(context, dims.dims()==1, 
      errors::InvalidArgument("Probe3D expects dims to be of dimension 1"));
    OP_REQUIRES(context, steps.dims()==1, 
      errors::InvalidArgument("Probe3D expects steps to be of dimension 1"));

    // Divide dims by strides to get steps.
    Tensor strides = (dims / steps).cast<int>();

    // Create an output tensor with the correct output shape
    // [num_batches, num_filters, x_steps, y_steps, z_steps]
    Tensor* output_tensor = NULL;
    int b = input_tensor.shape().dim_size(0);
    int f = weights.shape().dim_size(0);
    int x = steps.shape().dim_size(0);
    int y = steps.shape().dim_size(1);
    int z = steps.shape().dim_size(2);
    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{b,f,x,y,z},
                                                     &output_tensor));
    
    Probe3DFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        weights.flat<T>().data(),
        dims.data(),
        steps.data().
        output_tensor.flat<T>().data());
  }
};
REGISTER_KERNEL_BUILDER(Name("Probe3D").Device(DEVICE_GPU), Probe3DGpuOp);

#ifdef GOOGLE_CUDA
#define REGISTER_GPU(T)
  REGISTER_KERNEL_BUILDER(
    Name("Probe3D").Device(DEVICE_GPU).TypeConstraint<T>("T"), 
    Probe3DGpuOp<GPUDevice, T>);
REGISTER_GPU(float);
#endif // GOOGLE_CUDA
