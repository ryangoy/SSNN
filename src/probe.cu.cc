#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cstdlib>

// Takes in gridlist datastructure and runs K-means per sample for each step
__global__ void ProbeKernel(int batches, int filters, int probes_per_filter, int points, 
    float* gl_points, float* gl_indices, const float* weights, float xdim, float ydim, float zdim, int steps, float* output) {

    

    // N threads
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batches*steps*steps*steps*filters*probes_per_filter; i+= blockDim.x * gridDim.x){
        // Compute the sample the thread corresponds to. Could have used CUDA dim3, but we need to loop through more for loops than 3.
        int batch = i / (steps*steps*steps*filters*probes_per_filter) ;
        int x_step = i % (steps*steps*steps*filters*probes_per_filter) / (steps*steps*filters*probes_per_filter);
        int y_step = i % (steps*steps*filters*probes_per_filter) / (steps*filters*probes_per_filter);
        int z_step = i % (steps*filters*probes_per_filter) / (filters*probes_per_filter);
        int filter_id = i % (filters*probes_per_filter) / probes_per_filter;
        int probe_id = i % probes_per_filter;

        // Get the query index of the gridlist
        int start_index = gl_indices[batch*steps*steps*steps+x_step*steps*steps+y_step*steps+z_step];
        int end_index = gl_indices[batch*steps*steps*steps+x_step*steps*steps+y_step*steps+z_step+1];
        int num_vox_points = end_index - start_index;
        float* vox_gl_points = gl_points+start_index;

        // Corner of the voxel box
        float xc = xdim / steps * x_step;
        float yc = ydim / steps * y_step;
        float zc = zdim / steps * z_step;

        // Weight dims (this is coordinates of the probes, not the dot product weights)
        // Shape: (filters, probes_per_filter, xyz)
        int sample_index = filter_id*probes_per_filter*3 + probe_id*3;
        float sample_coord []= {weights[sample_index] + xc,
                                weights[sample_index+1] + yc,
                                weights[sample_index+2] + zc};
        float closest_dist = 100.0;

        // Loop through each point to find the nearest distance
        for (int point_index = 0; point_index < num_vox_points; point_index+=3) {
            float curr_point [] = {vox_gl_points[point_index], vox_gl_points[point_index+1],
                                   vox_gl_points[point_index+2]};
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

        // Add closest_dist to output
        output[batch*steps*steps*steps*filters*probes_per_filter
              +x_step*steps*steps*filters*probes_per_filter
              +y_step*steps*filters*probes_per_filter
              +z_step*filters*probes_per_filter
              +filter_id*probes_per_filter
              +probe_id] = closest_dist;
    }

}

__global__ void GenerateGridListIndices(int batches, int points, int steps, float x_step_size, float y_step_size, 
                                 float z_step_size, const float* pointcloud, float* output_indices, 
                                 float* output_points) {

    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < batches*steps*steps*steps; n+= blockDim.x * gridDim.x){
    //for (int n = 0; n < batches*steps*steps*steps; n++) {
        int grid_index = 0;
        int b = n / (steps*steps*steps);
        int i = n % (steps*steps*steps) / (steps*steps);
        int j = n % (steps*steps) / steps;
        int k = n % steps;
        
        // Compute bounds for the given voxel
        float x_min = i*x_step_size;
        float x_max = (i+1)*x_step_size;
        float y_min = j*y_step_size;
        float y_max = (j+1)*y_step_size;
        float z_min = k*z_step_size;
        float z_max = (k+1)*z_step_size;

        // For each point, add it to the voxel if it's within range
        for (int p = 0; p < points; p++) {

            float x_val = pointcloud[b*points*3+p*3];
            float y_val = pointcloud[b*points*3+p*3+1];
            float z_val = pointcloud[b*points*3+p*3+2];

            if (x_val >= x_min and x_val < x_max and 
                y_val >= y_min and y_val < y_max and 
                z_val >= z_min and z_val < z_max){

                grid_index += 1;
            }
        }
        output_indices[b*steps*steps*steps+i*steps*steps+j*steps+k] = grid_index*3;
    }
}

__global__ void GenerateGridList(int batches, int points, int steps, float x_step_size, float y_step_size, 
                                 float z_step_size, const float* pointcloud, float* output_indices, 
                                 float* output_points) {
    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < batches*steps*steps*steps; n+= blockDim.x * gridDim.x){
    //for (int n = 0; n < batches*steps*steps*steps; n++) {
        int b = n / (steps*steps*steps);
        int i = n % (steps*steps*steps) / (steps*steps);
        int j = n % (steps*steps) / steps;
        int k = n % steps;
        
        // Compute bounds for the given voxel
        float x_min = i*x_step_size;
        float x_max = (i+1)*x_step_size;
        float y_min = j*y_step_size;
        float y_max = (j+1)*y_step_size;
        float z_min = k*z_step_size;
        float z_max = (k+1)*z_step_size;

        int grid_index = output_indices[b*steps*steps*steps+i*steps*steps+j*steps+k];

        // For each point, add it to the voxel if it's within range
        for (int p = 0; p < points; p++) {

            float x_val = pointcloud[b*points*3+p*3];
            float y_val = pointcloud[b*points*3+p*3+1];
            float z_val = pointcloud[b*points*3+p*3+2];

            if (x_val >= x_min and x_val < x_max and 
                y_val >= y_min and y_val < y_max and 
                z_val >= z_min and z_val < z_max){

                output_points[grid_index*3] = x_val;
                output_points[grid_index*3+1] = y_val;
                output_points[grid_index*3+2] = z_val;
                grid_index += 1;
            }
        }
    }
}

void probeLauncher(int batches, int filters, int probes_per_filter, int points, const float* input_tensor, const float* weights,
      float xdim, float ydim, float zdim, int steps, float* output_tensor){

    // nb is the number of SM's we want to use
    // threads_per_block are number of threads per SM
    int nb = 32;
    int threads_per_block = 256;

    float x_step_size = xdim / steps;
    float y_step_size = ydim / steps;
    float z_step_size = zdim / steps;
    float* gl_indices;
    float* gl_points;

    // printf("batches: %d, filters: %d, probes_per_filter: %d, points: %d, xdim: %f, ydim: %f, zdim: %f, steps: %d\n", batches, filters, probes_per_filter,points, xdim, ydim, zdim, steps);

    // Allocate arrays for gridlist
    cudaMallocManaged(&gl_indices, batches*steps*steps*steps*sizeof(int));
    cudaMallocManaged(&gl_points, batches*points*3*sizeof(float));

    //printf("[Probe] CUDA arrays successfully allocated.\n");

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    GenerateGridListIndices<<<nb, threads_per_block>>>(batches, points, steps, x_step_size, y_step_size, z_step_size, 
                        input_tensor, gl_indices, gl_points);

    int prev_index = 0;
    int grid_index = gl_indices[0];
    gl_indices[0] = 0;
    for (int i = 1; i < batches*steps*steps*steps; i++) {
        prev_index = grid_index;
        grid_index += gl_indices[i];
        gl_indices[i] = prev_index; 
    }

    GenerateGridList<<<nb, threads_per_block>>>(batches, points, steps, x_step_size, y_step_size, z_step_size, 
                        input_tensor, gl_indices, gl_points);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    //printf("[Probe] Milliseconds to run gridlist datastructure: %f \n", milliseconds);

    cudaEvent_t start_2, stop_2;
    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);
    cudaEventRecord(start_2);
    ProbeKernel<<<nb, threads_per_block>>>
        (batches, filters, probes_per_filter, points, gl_points, gl_indices, weights, xdim, ydim, zdim, steps, output_tensor);
    cudaEventRecord(stop_2);
    cudaEventSynchronize(stop_2);
    float milliseconds_2 = 0;
    cudaEventElapsedTime(&milliseconds_2, start_2, stop_2);
    //printf("[Probe] Milliseconds to run probe filter: %f \n", milliseconds_2);

    cudaFree(gl_indices);
    cudaFree(gl_points);
    //printf("[Probe] Freed gridlist datastructure.\n");
}
