#ifdef GOOGLE_CUDA

#define EIGEN_USE_GPU
#define EIGEN_USE_THREADS

#include "probe_3d.h"

#include "tensorflow/core/util/cuda_kernel_helper.h"

using namespace tensorflow;

#define EIGEN_USE_GPU

template <typename T>
__global__ void Probe3DKernel(const int* sizes, const T* input, const T* weights, const T* dims, const T* steps, T* output) {
	// PSEUDO CODE
    // input: point cloud with size [n, p, 3]
    //        weights with size [n, c, 3]

    // for each interval in 3d_space:
    //   for each filter and probe:
    //     query points += interval_coord + xyz
    // return knn(query_points, point_cloud)

    // output: filter response with size [n, c, steps_x, steps_y, steps_z]
	// loop
	int x_steps = (int)steps[0];
	int y_steps = (int)steps[1];
	int z_steps = (int)steps[2];
	for (int batch = 0; batch < sizes[0]; batch += 1){
		int num_intervals = steps[0] * steps[1] * steps[2];
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
			float x = dims[0] / x_steps * x_step;
			float y = dims[1] / y_steps * y_step;
			float z = dims[2] / z_steps * z_step;

	    	// This would be the place to query an octree. Works well since
	    	// we won't be modifying the octree once it's created for a sample.
	    	for (int probe_id = 0; probe_id < sizes[1]; probe_id += 1) {
	    		int probe_index = batch*sizes[0]*3 + probe_id*3;
	    		T curr_probe []= {weights[probe_index],
	    						  weights[probe_index+1],
	    						  weights[probe_index+2]};

	    		// Compute the probe position in world coordinates.
	    		curr_probe[0] += x;
	    		curr_probe[1] += y;
	    		curr_probe[2] += z;

	    		// Init with max distance.
	    		float closest_x = 0.0f;
	    		float closest_y = 0.0f;
	    		float closest_z = 0.0f;
	    		float closest_dist = 1e12;
	    		for (int point_index = 0; point_index < sizes[2]; point_index += 1) {
					int curr_point_index = batch*sizes[2]*3+point_index*3;
					T curr_point [] = {input[curr_point_index], input[curr_point_index+1],
									 input[curr_point_index+2]};
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
	    		int output_index = batch*sizes[1]*num_intervals + probe_id*num_intervals + i;
	    		output[output_index] = closest_dist;
	    	}
	  	}
	}
}

template <typename T>
struct Probe3DFunctor<GPUDevice, T> {
	void operator()(const GPUDevice& d, const int* sizes, const T* input, 
  				  const T* weights, const T* dims, const T* steps, T* output) {
		// Launch cuda kernel
		// see core/util/cuda_kernel_helper.h to compute block count and
		// thread per block count
		int block_count = 32;
		int thread_per_block = 512;

		Probe3DKernel<T>
		<<<block_count, thread_per_block, 0, d.stream()>>>(sizes, input, weights, dims, steps, output);

	}
};

typedef Eigen::GpuDevice GPUDevice;
template struct Probe3DFunctor<GPUDevice, float>;

#endif // GOOGLE_CUDA