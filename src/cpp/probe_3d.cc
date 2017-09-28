#define USE_EIGEN_TENSOR
#define EIGEN_USE_THREADS

//#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
// #include "tensorflow/core/framework/register_types.h"
// #include "tensorflow/core/framework/tensor.h"
// #include "tensorflow/core/framework/tensor_shape.h"
// #include "tensorflow/core/framework/tensor_slice.h"
// #include "tensorflow/core/kernels/conv_ops_gpu.h"
// #include "tensorflow/core/kernels/ops_util.h"
// #include "tensorflow/core/lib/core/errors.h"
// #include "tensorflow/core/util/padding.h"
// #include "tensorflow/core/util/tensor_format.h"
// #include "tensorflow/core/util/use_cudnn.h"

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

class Probe3DOp : public OpKernel {
public:
  explicit ZeroOutOp(OpKernelConstruction* context) : OpKernel(context) {}

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

    // Divide dims by strides to get steps.
    /*********/
    /* TO DO */
    /*********/

    // Create an output tensor with the correct output shape
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, /* TO DO */,
                                                     &output_tensor));
    
    // Create a tensor with the coordinates of all the probes.
    /*********/
    /* TO DO */
    /*********/

    // Call PCL's implementation of KNN using octrees.
    

  }
};

// // Define Probe 3D kernel.
// typedef Eigen::ThreadPoolDevice CPUDevice;
// typedef Eigen::GpuDevice GPUDevice;

// // CPU implementation.
// template <typename Device, typename T>
// struct LaunchConvOp;

// template <typename T>
// struct LaunchConvOp<CPUDevice, T> {
//   static void launch(OpKernelContext* context, bool cudnn_use_autotune,
//                      const Tensor& input, const Tensor& filter,
//                      const std::array<int64, 3>& strides, const Padding padding,
//                      TensorFormat data_format, Tensor* output) {
//     OP_REQUIRES(context, data_format == FORMAT_NHWC,
//                 errors::InvalidArgument("CPU implementation of Probe3D "
//                                         "currently only supports the NHWC "
//                                         "tensor format."));
//     functor::CuboidProbing<CPUDevice, T>()(
//         context->eigen_device<CPUDevice>(), output->tensor<T, 5>(),
//         input.tensor<T, 5>(), filter.tensor<T, 5>(), strides[2], strides[1],
//         strides[0], BrainPadding2EigenPadding(padding));
//   }
// };

// template <typename Device, typename T>
// class Probe3D : public BinaryOp<T> {
// //class Probe3D : public OpKernel {
// public:
//   explicit Probe3DOp(OpKernelConstruction* context) : OpKernel(context) {
//     string data_format;
//     OP_REQUIRES_OK(context, context->GetAttr("data_format", &data_format));
//     OP_REQUIRES(context, FormatFromString(data_format, &data_format_),
//                 errors::InvalidArgument("Invalid data format"));
//     OP_REQUIRES_OK(context, context->GetAttr("strides", &stride_));
//     OP_REQUIRES(context, stride_.size() == 5,
//                 errors::InvalidArgument("Sliding window strides field must "
//                                         "specify 5 dimensions"));
//     OP_REQUIRES(
//         context,
//         (GetTensorDim(stride_, data_format_, 'N') == 1 &&
//          GetTensorDim(stride_, data_format_, 'C') == 1),
//         errors::InvalidArgument("Current implementation does not yet support "
//                                 "strides in the batch and depth dimensions."));
//     OP_REQUIRES_OK(context, context->GetAttr("padding", &padding_));
//     cudnn_use_autotune_ = CudnnUseAutotune();
//   }

//   void Compute(OpKernelContext* context) override {
//     DCHECK_EQ(2, context->num_inputs());


//     const Tensor& input = context->input(0);

//     // Input filter is of the following dimensions:
//     // [ filter_z, filter_y, filter_x, in_channels, out_channels]
//     const Tensor& filter = context->input(1);

//     // NOTE: The ordering of the spatial dimensions is arbitrary, but has to be
//     // kept consistent between input/filter/output.
//     OP_REQUIRES(context, input.dims() == 5,
//                 errors::InvalidArgument("input must be 5-dimensional"));
//     OP_REQUIRES(context, filter.dims() == 5,
//                 errors::InvalidArgument("filter must be 5-dimensional"));

//     const int64 in_depth = GetTensorDim(input, data_format_, 'C');
//     const int64 in_batch = GetTensorDim(input, data_format_, 'N');

//     const int64 out_depth = filter.dim_size(4);
//     OP_REQUIRES(
//         context, in_depth == filter.dim_size(3),
//         errors::InvalidArgument("input and filter must have the same depth"));

//     // Dimension order for these arrays is: z, y, x.
//     std::array<int64, 3> input_size = {
//         {GetTensorDim(input, data_format_, '0'),
//          GetTensorDim(input, data_format_, '1'),
//          GetTensorDim(input, data_format_, '2')}};
//     std::array<int64, 3> filter_size = {
//         {filter.dim_size(0), filter.dim_size(1), filter.dim_size(2)}};
//     std::array<int64, 3> strides = {{GetTensorDim(stride_, data_format_, '0'),
//                                      GetTensorDim(stride_, data_format_, '1'),
//                                      GetTensorDim(stride_, data_format_, '2')}};
//     std::array<int64, 3> out, padding;

//     OP_REQUIRES_OK(context, Get3dOutputSize(input_size, filter_size, strides,
//                                             padding_, &out, &padding));
//     TensorShape out_shape = ShapeFromFormat(
//         data_format_, in_batch, {{out[0], out[1], out[2]}}, out_depth);
//     Tensor* output;
//     OP_REQUIRES_OK(context, context->allocate_output(0, out_shape, &output));

//     // Return early if nothing to do.
//     if (out_shape.num_elements() == 0) return;

//     LaunchConvOp<Device, T>::launch(context, cudnn_use_autotune_, input, filter,
//                                     strides, padding_, data_format_, output);
//   }

//  private:
//   std::vector<int32> stride_;
//   Padding padding_;
//   TensorFormat data_format_;
//   bool cudnn_use_autotune_;
// };

// REGISTER_KERNEL_BUILDER(Name("Probe3D").Device(DEVICE_CPU), Probe3DOp);