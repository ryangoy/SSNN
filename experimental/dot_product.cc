#define EIGEN_USE_THREADS

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/op.h"

using namespace tensorflow;

// Define Probe interface.
REGISTER_OP("DotProduct")
  .Input("input: float32")
  .Input("weights: float32")
  .Output("output: float32");

void dotProductLauncher(int batches, int kernels, int samples_per_probe, int x_steps, int y_steps, int z_steps,
                        const float* input_tensor, const float* weights, float* output_tensor);
class DotProductOp : public OpKernel {
public:
  explicit DotProductOp(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("steps", &steps_));
  }

  void Compute(OpKernelContext* context) override {

    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& weights = context->input(1);
    OP_REQUIRES(context, input_tensor.dims()==6, 
      errors::InvalidArgument("Probe expects (batches, probes, samples_per_probe, x, y, z) input shape"));
    OP_REQUIRES(context, weights.dims()==2, 
      errors::InvalidArgument("Probe expects (probes, samples_per_probe) weights shape"));

    int steps = steps_;
    Tensor* output_tensor = NULL;
    int nbatches = input_tensor.shape().dim_size(0);
    int nkernels = weights.shape().dim_size(0);
    int nsamples = weights.shape().dim_size(1);
    int x_steps = input_tensor.shape().dim_size(3);
    int y_steps = input_tensor.shape().dim_size(4);
    int z_steps = input_tensor.shape().dim_size(5);


    OP_REQUIRES_OK(context, context->allocate_output(0, 
      TensorShape{nbatches,nkernels,x_steps,y_steps,z_steps}, &output_tensor));

    const float* inp = &(input_tensor.flat<float>()(0));
    float* out = &(output_tensor->flat<float>()(0));
    const float* cweights = &(weights.flat<float>()(0));
    
    probeLauncher(nbatches, nkernels, nsamples, x_steps, y_steps, z_steps, inp, cweights, out);
  }
};

REGISTER_KERNEL_BUILDER(Name("DotProduct").Device(DEVICE_GPU), DotProductOp);


