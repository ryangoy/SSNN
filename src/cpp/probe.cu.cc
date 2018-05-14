#include <stdio.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cstdlib>

/* Takes in gridlist datastructure and runs K-means per sample for each step */
__global__ void ProbeKernel(int batches, int filters, int probes_per_filter, int points, 
    float* gl_points, int* gl_indices, const float* weights, float xdim, float ydim, float zdim, int xy_steps, int z_steps, float ksize, float* output) {

    // N threads
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < batches*xy_steps*xy_steps*z_steps*filters*probes_per_filter; i+= blockDim.x * gridDim.x){
        // Compute the sample the thread corresponds to. Could have used CUDA dim3, but we need to loop through more for loops than 3.
        int batch = i / (xy_steps*xy_steps*z_steps*filters*probes_per_filter);
        int x_step = i % (xy_steps*xy_steps*z_steps*filters*probes_per_filter) / (xy_steps*z_steps*filters*probes_per_filter);
        int y_step = i % (xy_steps*z_steps*filters*probes_per_filter) / (z_steps*filters*probes_per_filter);
        int z_step = i % (z_steps*filters*probes_per_filter) / (filters*probes_per_filter);
        int filter_id = i % (filters*probes_per_filter) / probes_per_filter;
        int probe_id = i % probes_per_filter;

        // Get the query index of the gridlist. Offsets are computed for cases where points of kernels are outside the
        // current grid.
        int sample_index = filter_id*probes_per_filter*6 + probe_id*6;

        int x_offset = max(-x_step, min(xy_steps-1-x_step, int(floor(weights[sample_index]/ksize))));
        int y_offset = max(-y_step, min(xy_steps-1-y_step, int(floor(weights[sample_index+1]/ksize))));
        int z_offset = max(-z_step, min(z_steps-1-z_step, int(floor(weights[sample_index+2]/ksize))));

        // if (!printed && (x_offset != 0 || y_offset!= 0 || z_offset!=0)){
        //     printf("[DEBUG] a %i %i %i\n [DEBUG] b %f %f %f %f %i %i \n", x_offset, y_offset, z_offset, weights[sample_index], weights[sample_index+1], weights[sample_index+2], ksize, -x_step, steps-1-x_step);
        //     printed = true;

        // }

        int start_index_index = batch*xy_steps*xy_steps*z_steps+(x_step+x_offset)*xy_steps*z_steps+(y_step+y_offset)*z_steps+z_step+z_offset;

        // Access the proper grid.
        int start_index = gl_indices[start_index_index];
        int end_index = gl_indices[start_index_index+1];
        int num_vox_points = end_index - start_index;
        float* vox_gl_points = gl_points+start_index;

        // Compute the corner of the voxel box.
        float xc = xdim / xy_steps * x_step;
        float yc = ydim / xy_steps * y_step;
        float zc = zdim / z_steps * z_step;

        // Weight dims (this is coordinates of the probes, not the dot product weights)
        // Shape: (filters, probes_per_filter, xyz)
        
        float sample_coord []= {weights[sample_index] + xc,
                                weights[sample_index+1] + yc,
                                weights[sample_index+2] + zc};
        float closest_dist = 1.1;
        float closest_r = 0.0;
        float closest_g = 0.0;
        float closest_b = 0.0;


        // Loop through each point to find the nearest distance
        for (int point_index = 0; point_index < num_vox_points; point_index+=6) {
            float curr_point [] = {vox_gl_points[point_index], vox_gl_points[point_index+1],
                                   vox_gl_points[point_index+2]};
            float dist = (sample_coord[0]-curr_point[0])*(sample_coord[0]-curr_point[0]) 
                        +(sample_coord[1]-curr_point[1])*(sample_coord[1]-curr_point[1]) 
                        +(sample_coord[2]-curr_point[2])*(sample_coord[2]-curr_point[2]);
            if (dist < closest_dist) {
                closest_dist = dist;
                closest_r = vox_gl_points[point_index+3];
                closest_g = vox_gl_points[point_index+4];
                closest_b = vox_gl_points[point_index+5];
            } 
        }

        // For a higher response for closer points, we take the negative. 0.1 is a hard-coded value for now so that
        // the response's mean is close to 0, but it doesn't matter too much.
        closest_dist = 0.1-closest_dist;

        // Add closest_dist to output
        int curr_output_index = batch*xy_steps*xy_steps*z_steps*filters*probes_per_filter*4
              +x_step*xy_steps*z_steps*filters*probes_per_filter*4
              +y_step*z_steps*filters*probes_per_filter*4
              +z_step*filters*probes_per_filter*4
              +filter_id*probes_per_filter*4
              +probe_id*4;
        output[curr_output_index] = closest_dist;
        output[curr_output_index+1] = closest_r;
        output[curr_output_index+2] = closest_g;
        output[curr_output_index+3] = closest_b;

    }

}

/* Computes number of voxels per grid cell. */
__global__ void GenerateGridListIndices(int batches, int points, int xy_steps, int z_steps, float x_step_size, float y_step_size, 
                                 float z_step_size, const float* pointcloud, int* output_indices, 
                                 float* output_points) {

    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < batches*xy_steps*xy_steps*z_steps; n+= blockDim.x * gridDim.x){
        
        // Calculate the specific output value to compute.
        int grid_index = 0;
        int b = n / (xy_steps*xy_steps*z_steps);
        int i = n % (xy_steps*xy_steps*z_steps) / (xy_steps*z_steps);
        int j = n % (xy_steps*xy_steps) / z_steps;
        int k = n % z_steps;
        
        // Compute bounds for the given voxel.
        float x_min = i*x_step_size;
        float x_max = (i+1)*x_step_size;
        float y_min = j*y_step_size;
        float y_max = (j+1)*y_step_size;
        float z_min = k*z_step_size;
        float z_max = (k+1)*z_step_size;

        // For each point, add it to the voxel if it's within range.
        for (int p = 0; p < points; p++) {

            float x_val = pointcloud[b*points*6+p*6];
            float y_val = pointcloud[b*points*6+p*6+1];
            float z_val = pointcloud[b*points*6+p*6+2];

            if (x_val >= x_min and x_val < x_max and 
                y_val >= y_min and y_val < y_max and 
                z_val >= z_min and z_val < z_max){

                grid_index += 1;
            }
        }

        // Set the number of voxels values needed for a specific grid list.
        output_indices[b*xy_steps*xy_steps*z_steps+i*xy_steps*z_steps+j*z_steps+k] = grid_index*6;
    }
}

/* From the output of GenerateGridListIndices, this computes the starting index of the final output for each grid cell.*/
__global__ void ArrangeGridListIndices(int batches, int xy_steps, int z_steps, int* gl_indices) {

    int prev_index = 0;
    int grid_index = gl_indices[0];
    gl_indices[0] = prev_index;
    for (int i = 1; i < batches*xy_steps*xy_steps*z_steps; i++) {
        prev_index = grid_index;
        grid_index += gl_indices[i];
        
        gl_indices[i] = prev_index; 
    }
}

/* Puts points in grid cells based on index from ArrangeGridListIndices. */
__global__ void GenerateGridList(int batches, int points, int xy_steps, int z_steps, float x_step_size, float y_step_size, 
                                 float z_step_size, const float* pointcloud, int* output_indices, 
                                 float* output_points) {

    for (int n = blockIdx.x * blockDim.x + threadIdx.x; n < batches*xy_steps*xy_steps*z_steps; n+= blockDim.x * gridDim.x){
    //for (int n = 0; n < batches*steps*steps*steps; n++) {
        
        int b = n / (xy_steps*xy_steps*z_steps);
        int i = n % (xy_steps*xy_steps*z_steps) / (xy_steps*z_steps);
        int j = n % (xy_steps*xy_steps) / z_steps;
        int k = n % z_steps;
        
        // Compute bounds for the given voxel
        float x_min = i*x_step_size;
        float x_max = (i+1)*x_step_size;
        float y_min = j*y_step_size;
        float y_max = (j+1)*y_step_size;
        float z_min = k*z_step_size;
        float z_max = (k+1)*z_step_size;
        int grid_index = output_indices[b*xy_steps*xy_steps*z_steps+i*xy_steps*z_steps+j*z_steps+k];
        //printf("iter: %d, grid_index: %d\n", n, grid_index);
        // For each point, add it to the voxel if it's within range
        for (int p = 0; p < points; p++) {

            float x_val = pointcloud[b*points*6+p*6];
            float y_val = pointcloud[b*points*6+p*6+1];
            float z_val = pointcloud[b*points*6+p*6+2];
            float r_val = pointcloud[b*points*6+p*6+3];
            float g_val = pointcloud[b*points*6+p*6+4];
            float b_val = pointcloud[b*points*6+p*6+5];

            if (x_val >= x_min and x_val < x_max and 
                y_val >= y_min and y_val < y_max and 
                z_val >= z_min and z_val < z_max){

                output_points[grid_index] = x_val;
                output_points[grid_index+1] = y_val;
                output_points[grid_index+2] = z_val;
                output_points[grid_index+3] = r_val;
                output_points[grid_index+4] = g_val;
                output_points[grid_index+5] = b_val;

                grid_index += 6;
            }
        }
    }
}

void probeLauncher(int batches, int filters, int probes_per_filter, int points, const float* input_tensor, const float* weights,
      float xdim, float ydim, float zdim, int xy_steps, int z_steps, float ksize, float* output_tensor){

    // nb is the number of SM's we want to use
    // threads_per_block are number of threads per SM
    int nb = 64;
    int threads_per_block = 512;

    float x_step_size = xdim / xy_steps;
    float y_step_size = ydim / xy_steps;
    float z_step_size = zdim / z_steps;
    int* gl_indices;
    float* gl_points;

    // Allocate arrays for gridlist
    cudaMallocManaged(&gl_indices, batches*xy_steps*xy_steps*z_steps*sizeof(int));
    cudaMallocManaged(&gl_points, batches*points*6*sizeof(float));

    /*** Generate list indices ***/
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    GenerateGridListIndices<<<nb, threads_per_block>>>(batches, points, xy_steps, z_steps, x_step_size, y_step_size, z_step_size, 
                        input_tensor, gl_indices, gl_points);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    /*** Arrange list indices ***/
    cudaEvent_t start_2, stop_2;
    cudaEventCreate(&start_2);
    cudaEventCreate(&stop_2);
    cudaEventRecord(start_2);
    ArrangeGridListIndices<<<1, 1>>>(batches, xy_steps, z_steps, gl_indices);
    cudaEventRecord(stop_2);
    cudaEventSynchronize(stop_2);
    float milliseconds_2 = 0;
    cudaEventElapsedTime(&milliseconds_2, start_2, stop_2);

    /*** Arrange list indices ***/
    cudaEvent_t start_3, stop_3;
    cudaEventCreate(&start_3);
    cudaEventCreate(&stop_3);
    cudaEventRecord(start_3);
    GenerateGridList<<<nb, threads_per_block>>>(batches, points, xy_steps, z_steps, x_step_size, y_step_size, z_step_size, 
                        input_tensor, gl_indices, gl_points);
    cudaEventRecord(stop_3);
    cudaEventSynchronize(stop_3);
    float milliseconds_3 = 0;
    cudaEventElapsedTime(&milliseconds_3, start_3, stop_3);

    /*** Run probing ***/
    cudaEvent_t start_4, stop_4;
    cudaEventCreate(&start_4);
    cudaEventCreate(&stop_4);
    cudaEventRecord(start_4);
    ProbeKernel<<<nb, threads_per_block>>>
        (batches, filters, probes_per_filter, points, gl_points, gl_indices, weights, xdim, ydim, zdim, xy_steps, z_steps, ksize, output_tensor);
    cudaEventRecord(stop_4);
    cudaEventSynchronize(stop_4);
    float milliseconds_4 = 0;
    cudaEventElapsedTime(&milliseconds_4, start_4, stop_4);

    // Free GPU memory.
    cudaFree(gl_indices);
    cudaFree(gl_points);
}