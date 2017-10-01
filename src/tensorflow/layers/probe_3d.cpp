#define EIGEN_USE_THREADS

#include "probe_3d.h"
#include "tensorflow/core/framework/op_kernel.h"

// #if GOOGLE_CUDA
// #include "tensorflow/core/platform/stream_executor.h"
// using perftools::gputools::dnn::DimIndex;
// #endif

using namespace tensorflow;

// Define Probe3D interface.
REGISTER_OP("Probe3D")
  .Input("input: float")
  .Input("weights: float")
  .Output("probe_3d: float")
  .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
    // shape_inference::ShapeHandle input_shape;
    // TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 2, &input_shape));

    // shape_inference::ShapeHandle weight_shape;
    // TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &weight_shape));
    
    // shape_inference::DimensionHandle output_rows = c->Dim(weight_shape, 0);
  
    // shape_inference::DimensionHandle input_rows = c->Dim(input_shape, 0);
    // shape_inference::DimensionHandle weight_cols = c->Dim(weight_shape, 1);
    // shape_inference::DimensionHandle merged;
    // TF_RETURN_IF_ERROR(c->Merge(input_rows, weight_cols, &merged));

    // c->set_output(0, c->Matrix(output_rows, 1));
    return Status::OK();
  });



void Probe3DLauncher(input_tensor, weights, num_steps, strides);
class Probe3DGpuOp : public OpKernel {
public:
  explicit Probe3DGpuOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // PSEUDO CODE
    // input: filters with size [n, x, y, z, c]

    // for each interval in 3d_space:
    //   for each filter and probe:
    //     query points += interval_coord + xyz
    // return knn(query_points, point_cloud)

    // output: filter response with size [n, steps_x, steps_y, steps_z, c]

    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& weights = context->input(1);
    const Tensor& dims = context->input(2);
    const Tensor& strides = context->input(3);

    OP_REQUIRES(context, input_tensor.dims()==5, 
      errors::InvalidArgument("Probe3D expects (n, p, x, y, z) input shape"));
    OP_REQUIRES(context, weights.dims()==5, 
      errors::InvalidArgument("Probe3D expects (n, x, y, z, c) weights shape"));
    OP_REQUIRES(context, dims.dims()==1, 
      errors::InvalidArgument("Probe3D expects dims to be of dimension 1"));
    OP_REQUIRES(context, strides.dims()==1, 
      errors::InvalidArgument("Probe3D expects strides to be of dimension 1"));

    // Divide dims by strides to get steps.
    Tensor num_steps = (dims / strides).cast<int>();

    // Create an output tensor with the correct output shape
    // [n, steps_x, steps_y, steps_z, c]
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, /* TO DO */,
                                                     &output_tensor));
    
    // Create a tensor with the coordinates of all the probes.
    /*********/
    /* TO DO */
    /*********/
    Probe3DFunctor<Device, T>()(
        context->eigen_device<Device>(),
        static_cast<int>(input_tensor.NumElements()),
        input_tensor.flat<T>().data(),
        weights.flat<T>().data(),
        num_steps.data(),
        strides.data().
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
