# visualizes bboxes and associated classes for predictions or labels. plot_bboxes.py is a beter visualization per category

import numpy as np
import matplotlib.pyplot as plt
import sys
import matplotlib.patches as patches
from matplotlib.pyplot import text

bboxes = np.load(sys.argv[1])

cls_vals = None
if len(sys.argv) > 2:
  cls_vals = np.load(sys.argv[2])

grid_size = 7.5
num_steps = 16
kernel_size = grid_size/num_steps

for scene in range(bboxes.shape[0]):
  print('Scene {}:'.format(scene))
  f, axarr = plt.subplots(int(num_steps/4), 4)
  f.set_figwidth(grid_size)
  for z_dim in range(num_steps):
    for bb_id in range(len(bboxes[scene])):

      bbox = bboxes[scene][bb_id]
      if cls_vals is not None:
        cls_val = cls_vals[scene][bb_id]
        most_likely_class = cls_val.argmax()

      center_loc = (bbox[:3] + bbox[3:])/2
      bbox_size = bbox[3:] - bbox[:3]
      curr_z = z_dim * kernel_size
      axarr[int(z_dim/4), z_dim%4].set_xlim([0, grid_size])
      axarr[int(z_dim/4), z_dim%4].set_ylim([0, grid_size])   

      if (bbox[2] <= curr_z+kernel_size and bbox[2] >= curr_z) or\
         (bbox[5] <= curr_z+kernel_size and bbox[5] >= curr_z) or\
         (bbox[5] >= curr_z+kernel_size and bbox[2] <= curr_z):

        axarr[int(z_dim/4), z_dim%4].add_patch(patches.Rectangle(bbox[:2], bbox_size[0], bbox_size[1]))
        if cls_vals is not None:
          axarr[int(z_dim/4), z_dim%4].text(bbox[0], bbox[1], str(most_likely_class), color='orange')

  plt.show()
