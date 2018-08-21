# visualizes the cls (classification) voxelgrid of labels or predictions at all scales

import numpy as np
import matplotlib.pyplot as plt
import sys

def visualize_scales(voxels_path, n_steps, n_scale):
  voxels = np.load(voxels_path)
  NUM_STEPS = n_steps
  NUM_SCALE = n_scale

  num_graphs = 0
  s = NUM_STEPS
  for i in range(NUM_STEPS):
    num_graphs += s
    s /= 2

  for scene in range(voxels.shape[0]):
    print('Scene {}:'.format(scene))

    # uncomment
    f, axarr = plt.subplots(int(num_graphs/4), 4)

    last_index = 0
    curr_index = 0
    num_steps = NUM_STEPS

    for scale in range(NUM_SCALE):
      d_slice = voxels[scene, last_index:last_index+int(num_steps)**3, ..., 1:]
      d_slice = d_slice.sum(axis=-1)
      d_slice = np.reshape(d_slice, (num_steps, num_steps, num_steps, -1))

      d_slice = np.max(d_slice, axis=-1)
      # d_slice = d_slice[...,0]

      for z_dim in range(num_steps):
        #uncomment
        axarr[int(curr_index/4), curr_index%4].imshow(d_slice[:,:,z_dim].T, interpolation='nearest', cmap='hot', vmin=0, vmax=1)
        axarr[int(curr_index/4), curr_index%4].invert_yaxis()
        curr_index += 1
        
        #comment
        # if scale == NUM_SCALE-1 and z_dim == 0:
        #   plt.imshow(d_slice[:,:,z_dim].T, interpolation='nearest', cmap='hot', vmin=0, vmax=1)
        #   plt.xticks([0, 1, 2, 3])
        #   plt.yticks([0, 1, 2, 3])
          #finish comment
      last_index += num_steps**3
      num_steps = int(num_steps/2)

    plt.show()

if __name__ == '__main__':

  visualize_scales(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
