#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "probe_3d.hpp"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

template <typename T>
__global__ void Probe3DKernel(const T* input, 
  				  T* weights, T* dims, T* steps, T* output) {
	// PSEUDO CODE
    // input: point cloud with size [n, p, 3]
    //        weights with size [n, c, 3]

    // for each interval in 3d_space:
    //   for each filter and probe:
    //     query points += interval_coord + xyz
    // return knn(query_points, point_cloud)

    // output: filter response with size [n, steps_x, steps_y, steps_z, c]
	// loop
}

template <typename T>
struct Probe3DFunctor<GPUDevice, T> {

	void operator()(const GPUDevice& d, const T* input, 
  				  T* weights, T* dims, T* steps, T* output) {
		// Launch cuda kernel
		// see core/util/cuda_kernel_helper.h to compute block count and
		// thread per block count
		int block_count = 32;
		int thread_per_block = 512;
		Probe3DKernel<T>
		<<<block_count, thread_per_block, 0, d.stream()>>>(input, weights, dims, steps, output);

	}
};

typedef Eigen::GPUDevice GPUDevice;
template struct Probe3DFunctor<GPUDevice, float>;

#endif // GOOGLE_CUDA