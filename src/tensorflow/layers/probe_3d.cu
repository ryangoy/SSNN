#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "probe_3d.hpp"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

template <typename T>
__global__ void Probe3DKernel(const int size, const T* in, 
  				  T* weights, T* num_strides, T* out) {
	// loop
}

template <typename T>
struct Probe3DFunctor<GPUDevice, T> {
	void operator()(const GPUDevice& d, int size, const T* in, 
  				  T* weights, T* num_strides, T* out) {
		// Launch cuda kernel
		// see core/util/cuda_kernel_helper.h to compute block count and
		// thread per block count
		int block_count = 1024;
		int thread_per_block = 20;
		Probe3DKernel<T>
		<<<block_count, thread_per_block, 0, d.stream()>>>(size, in, weights, num_strides, out);

	}
};

typedef Eigen::GPUDevice GPUDevice;
template struct Probe3DFunctor<GPUDevice, float>;

#endif // GOOGLE_CUDA