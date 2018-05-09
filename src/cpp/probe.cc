#define EIGEN_USE_THREADS
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"
#include <stdio.h>
using namespace tensorflow;

// Define Probe interface.
REGISTER_OP("Probe")
  .Attr("xy_steps: int = 10")
  .Attr("z_steps: int=10")
  .Attr("xdim: float = 10.0")
  .Attr("ydim: float = 10.0")
  .Attr("zdim: float = 10.0")
  .Attr("ksize: float = 1.0")
  .Input("input: float32")
  .Input("weights: float32")
  .Output("output: float32");

// Boilerplate code for CUDA call
void probeLauncher(int batches, int kernels, int samples_per_probe, int points, const float* input_tensor, const float* weights,
      float xdim, float ydim, float zdim, int xy_steps, int z_steps, float ksize, float* output_tensor);
class ProbeOp : public OpKernel {
public:
  explicit ProbeOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("xy_steps", &xy_steps_));
    OP_REQUIRES_OK(context, context->GetAttr("z_steps", &z_steps_));
    OP_REQUIRES_OK(context, context->GetAttr("xdim", &xdim_));
    OP_REQUIRES_OK(context, context->GetAttr("ydim", &ydim_));
    OP_REQUIRES_OK(context, context->GetAttr("zdim", &zdim_));
    OP_REQUIRES_OK(context, context->GetAttr("ksize", &ksize_));
  }

  void Compute(OpKernelContext* context) override {
    // Fetch the input tensor and weights
    const Tensor& input_tensor = context->input(0);
    const Tensor& weights = context->input(1);
    OP_REQUIRES(context, input_tensor.dims()==3, 
      errors::InvalidArgument("Probe expects (batches, points, 6) input shape"));

    // Weight dims are similar to convolutional weights:
    //   Convolutional weights are (kernel_width, kernel_height, num_input_kernels, num_output_kernels)
    //   Dot product weights are (probe_id, num_input_kernels, num_output_kernels)
    OP_REQUIRES(context, weights.dims()==3, 
      errors::InvalidArgument("Probe expects (probes, kernels, output_features) weights shape"));

    // Fetch extra information (there's probably a better way to do this)
    int xy_steps = xy_steps_;
    int z_steps = z_steps_;
    float ksize = ksize_;
    float xdim = xdim_;
    float ydim = ydim_;
    float zdim = zdim_;

    // Create an output tensor with the correct output shape:
    //    (num_batches, x_steps, y_steps, z_steps, num_filters, num_samples)
    Tensor* output_tensor = NULL;
    int nbatches = input_tensor.shape().dim_size(0);
    int npoints = input_tensor.shape().dim_size(1);
    int nkernels = weights.shape().dim_size(0);
    int nsamples = weights.shape().dim_size(1);

    OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape{nbatches,xy_steps,xy_steps,z_steps,nkernels,nsamples,4},
                                                     &output_tensor));

    // Flatten inputs into 1D arrays to feed into CUDA code
    const float* inp = &(input_tensor.flat<float>()(0));
    float* out = &(output_tensor->flat<float>()(0));
    const float* cweights = &(weights.flat<float>()(0));
    probeLauncher(nbatches, nkernels, nsamples, npoints, inp, cweights, xdim, ydim, zdim, xy_steps, z_steps, ksize, out);
  }
private:
  int xy_steps_;
  int z_steps_;
  float ksize_;
  float xdim_;
  float ydim_;
  float zdim_;
};

// Register the kernel
REGISTER_KERNEL_BUILDER(Name("Probe").Device(DEVICE_GPU), ProbeOp);