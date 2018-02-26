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

  
    f, axarr = plt.subplots(num_graphs/4, 4)

    last_index = 0
    curr_index = 0
    num_steps = NUM_STEPS

    for scale in range(NUM_SCALE):
      d_slice = voxels[scene, last_index:last_index+num_steps**3, 1]
      d_slice = np.reshape(d_slice, (num_steps, num_steps, num_steps))

      for z_dim in range(num_steps):
        axarr[curr_index/4, curr_index%4].imshow(d_slice[:,:,z_dim].T, interpolation='nearest', cmap='hot', vmin=0, vmax=1)
        axarr[curr_index/4, curr_index%4].invert_yaxis()
        curr_index += 1

      last_index += num_steps**3
      num_steps /= 2

    plt.show()

if __name__ == '__main__':
  visualize_scales(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
