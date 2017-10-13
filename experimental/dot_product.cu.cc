#include <stdio.h>

__global__ void DotProductKernel(int batches, int probes, int samples_per_probe, int x_steps, int y_steps, int z_steps,
    const float* input_tensor, const float* weights, float* output_tensor) {
	// PSEUDO CODE
    // input: probe response with shape (batches, probes, samples_per_probe, x, y, z)
    //        weights with size [n, c, 3]

    // for each interval in 3d_space:
    //   for each filter and probe:
    //     query points += interval_coord + xyz
    // return knn(query_points, point_cloud)

    // output: filter response with size [n, c, steps_x, steps_y, steps_z]

    int num_intervals = x_steps * y_steps * z_steps;

    for (int batch = 0; batch < batches; batch++) {
        for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < x_steps; i+= blockDim.x * gridDim.x){
            for (int j = blockIdx.y * blockDim.y + threadIdx.y; j < y_steps; j+= blockDim.y * gridDim.y) {
                for (int k = blockIdx.z * blockDim.z + threadIdx.z; k < z_steps; k+= blockDim.z * gridDim.z) {
                    for (int probe_id = 0; probe_id < probes; probe_id++) {

                        int curr_dot_prod = 0;
                        for (int sample_id = 0; sample_id < samples_per_probe; sample_id++) {
                            curr_dot_prod += weights[probe_id, sample_id] * 
                            input_tensor[batch*probes*samples_per_probe*x_steps*y_steps*z_steps+
                                probe_id*samples_per_probe*x_steps*y_steps*z_steps+
                                sample_id*x_steps*y_steps*z_steps+
                                i*y_steps*z_steps+j*z_steps*k];
                        }
                        //nbatches,nkernels,x_steps,y_steps,z_steps
                        output_tensor[batch*probes*x_steps*y_steps*z_steps+
                                probe_id*x_steps*y_steps*z_steps+
                                i*y_steps*z_steps+j*z_steps*k];   
                    }
                }
            }
        }
    }
}

void dotProductLauncher(nbatches, nkernels, nsamples, x_steps, y_steps, z_steps, inp, cweights, out){
    int threads_per_block = 512;

    ProbeKernel<<<dim3(x_steps, y_steps, z_steps), threads_per_block>>>
        (nbatches, nkernels, nsamples, x_steps, y_steps, z_steps, input_tensor, weights, output_tensor);
}