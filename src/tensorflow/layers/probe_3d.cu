#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "probe_3d.hpp"
#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

template <typename T>
__global__ void Probe3DKernel(int b, int x_steps, int y_steps, int z_steps,
	int x_size, int y_size, int z_size, int num_probes, int num_points,
	const T* input, T* weights, T* dims, T* steps, T* output) {
	// PSEUDO CODE
    // input: point cloud with size [n, p, 3]
    //        weights with size [n, c, 3]

    // for each interval in 3d_space:
    //   for each filter and probe:
    //     query points += interval_coord + xyz
    // return knn(query_points, point_cloud)

    // output: filter response with size [n, c, steps_x, steps_y, steps_z]
	// loop
	for (int batch = 0; batch < b; batch += 1){}
		int num_intervals = x_steps * y_steps * z_steps;
		int yz_size = y_steps * z_steps;
		// blockIdx.x is the id of the block
		// blockDim.x is the number of threads for a block
		// threadIdx.x is the thread number of the block

		// Loop through each step
		for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num_intervals;
	      	i += blockDim.x * gridDim.x) {
			
			// Compute the step coordinate that i corresponds to.
			int x_step = (int)(i / yz_size);
			int y_step = (int)(i / z_steps) % y_steps;
			int z_step = i % z_steps;

			// Convert step coordinates to world coordinates. 
			float x = x_size / x_steps * x_step;
			float y = y_size / y_steps * y_step;
			float z = z_size / z_steps * z_step;

	    	// This would be the place to query an octree. Works well since
	    	// we won't be modifying the octree once it's created for a sample.
	    	for (int probe_id = 0; probe_id < num_probes; probe_id += 1) {
	    		T* curr_probe = weights[batch][probe_id];

	    		// Compute the probe position in world coordinates.
	    		curr_probe[0] += x;
	    		curr_probe[1] += y;
	    		curr_probe[2] += z;

	    		// Init with max distance.
	    		float closest_x = 0.0f;
	    		float closest_y = 0.0f;
	    		float closest_z = 0.0f;
	    		float closest_dist = FLT_MAX;
	    		for (int point_index = 0; point_index < num_points; point_index += 1) {
					T* curr_point = input[batch][point_index]
	    			float dist = (curr_point[0]-curr_probe[0])*(curr_point[0]-curr_probe[0]) 
	    						+(curr_point[1]-curr_probe[1])*(curr_point[1]-curr_probe[1]) 
	    						+(curr_point[2]-curr_probe[2])*(curr_point[2]-curr_probe[2]);
	    			if (dist < closest_dist) {
	    				closest_dist = dist;
	    				closest_x = curr_probe[0] - curr_point[0];
	    				closest_y = curr_probe[1] - curr_point[1];
	    				closest_z = curr_probe[2] - curr_point[1];
	    			} 
	    		}
	    		output[batch][probe_id][x_step][y_step][z_step] = closest_dist;
	    	}
	  	}
	}
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