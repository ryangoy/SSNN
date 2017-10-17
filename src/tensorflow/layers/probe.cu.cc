
#include <stdio.h>
#include <vector>
// #define EIGEN_USE_GPU
// #define EIGEN_USE_THREADS

// #include "probe.h"
// #include "tensorflow/core/util/cuda_kernel_helper.h"

// using namespace tensorflow;

// #define EIGEN_USE_GPU


__global__ void ProbeKernel(int batches, int filters, int samples_per_probe, int points, 
    const float* input, const float* weights, const float* dims, int steps, float* output) {
	// PSEUDO CODE
    // input: point cloud with size [n, p, 3]
    //        weights with size [n, c, 3]

    // for each interval in 3d_space:
    //   for each filter and probe:
    //     query points += interval_coord + xyz
    // return knn(query_points, point_cloud)

    // output: filter response with size [n, c, steps_x, steps_y, steps_z]

    int num_intervals = steps * steps * steps;

    for (int batch = 0; batch < batches; batch++) {
        for (int i = blockIdx.x; i < steps; i+= gridDim.x){
        //for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < steps; i+= blockDim.x * gridDim.x){
            for (int j = blockIdx.y; j < steps; j+= gridDim.y) {
            // for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < steps; j+= blockDim.y * gridDim.y) {
                for (int k = blockIdx.z; k < steps; k+= gridDim.z) {
                // for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < steps; k+= blockDim.z * gridDim.z) {
                    // Convert step coordinates to world coordinates. 
                    float xc = dims[0] / steps * i;
                    float yc = dims[1] / steps * j;
                    float zc = dims[2] / steps * k;
                    for (int probe_id = 0; probe_id < filters; probe_id ++) {
                        for (int sample_id = 0; sample_id < samples_per_probe; sample_id++) {
                            int sample_index = batch*filters*3 + probe_id*3;
                            float sample_coord []= {weights[sample_index] + xc,
                                                    weights[sample_index+1] + yc,
                                                    weights[sample_index+2] + zc};
                            float closest_dist = 1e38;
                            // This is where binned data would be called:
                            // printf("num points is %d\n", points);
                            for (int point_index = 0; point_index < points; point_index++) {
                                int curr_point_index = batch*points*3+point_index*3;
                                float curr_point [] = {input[curr_point_index], input[curr_point_index+1],
                                                       input[curr_point_index+2]};
                                float dist = (sample_coord[0]-curr_point[0])*(sample_coord[0]-curr_point[0]) 
                                            +(sample_coord[1]-curr_point[1])*(sample_coord[1]-curr_point[1]) 
                                            +(sample_coord[2]-curr_point[2])*(sample_coord[2]-curr_point[2]);
                                if (dist < closest_dist) {
                                    closest_dist = dist;
                                     // closest_x = curr_probe[0] - curr_point[0];
                                     // closest_y = curr_probe[1] - curr_point[1];
                                     // closest_z = curr_probe[2] - curr_point[1];
                                } 
                            }
                            output[batch*filters*samples_per_probe*num_intervals
                                +probe_id*samples_per_probe*num_intervals+sample_id*num_intervals
                                +i*steps*steps+j*steps+k] = closest_dist;
                        }    
                    }
                }
            }
        }
    }

}

__global__ void BinPointsKernel(int points, int step_size, const float* dims, const float* pointcloud, std::vector<float*>* output) {
    x_bin_size = dims[0] / step_size;
    y_bin_size = dims[1] / step_size;
    z_bin_size = dims[2] / step_size;

    for (int point_id=blockIdx.x * blockDim.x + threadIdx.x; point_id < points; point_id+= blockDim.x * gridDim.x) {
        int x_bin = (int)(pointcloud[point_id * 3] / x_bin_size); 
        int y_bin = (int)(pointcloud[point_id * 3 + 1] / y_bin_size); 
        int z_bin = (int)(pointcloud[point_id * 3 + 2] / z_bin_size); 

        float pt[3] = {pointcloud[point_id * 3], pointcloud[point_id * 3 + 1], pointcloud[point_id * 3 + 2]};
        output[x_bin*y_bin_size*z_bin_size + y_bin*z_bin_size + z_bin].push_back(pt);

    }
}

void probeLauncher(int batches, int filters, int samples_per_probe, int points, const float* input_tensor, const float* weights,
      const float* dims, int steps, float* output_tensor){
    int threads_per_block = 1;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    ProbeKernel<<<dim3(1, 1, 1), threads_per_block>>>
        (batches, filters, samples_per_probe, points, input_tensor, weights, dims, steps, output_tensor);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("milliseconds to run: %f \n", milliseconds);
}