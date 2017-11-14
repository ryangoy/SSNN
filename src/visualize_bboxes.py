import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as patches

bboxes = np.load(sys.argv[1])
grid_size = 7.5
num_steps = 16
kernel_size = grid_size/num_steps

for scene in range(bboxes.shape[0]):
  print('Scene {}:'.format(scene))
  f, axarr = plt.subplots(num_steps/4, 4, )

  f.set_figwidth(grid_size)
  for z_dim in range(num_steps):
    for bbox in bboxes[scene]:
      center_loc = (bbox[:3] + bbox[3:])/2
      bbox_size = bbox[3:] - bbox[:3]
      curr_z = z_dim * kernel_size
      axarr[z_dim/4, z_dim%4].set_xlim([0, grid_size])
      axarr[z_dim/4, z_dim%4].set_ylim([0, grid_size])      
      if (bbox[2] < curr_z+kernel_size and bbox[2] > curr_z) or\
         (bbox[5] < curr_z+kernel_size and bbox[5] > curr_z) or\
         (bbox[5] > curr_z+kernel_size and bbox[2] < curr_z):
        axarr[z_dim/4, z_dim%4].add_patch(patches.Rectangle(center_loc[:2], bbox_size[0], bbox_size[1]))
  plt.show()
