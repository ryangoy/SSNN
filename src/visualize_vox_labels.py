import numpy as np
import matplotlib.pyplot as plt
import sys

#voxels = np.load('probe_output.npy')
#voxels = np.load('vox_labels.npy')
#voxels = np.load('bboxes.npy')
#voxels = np.load('predictions.npy')
voxels = np.load(sys.argv[1])

for scene in range(voxels.shape[0]):
  print('Scene {}:'.format(scene))
  f, axarr = plt.subplots(voxels.shape[1]/4, 4)
  for z_dim in range(voxels.shape[1]):
    d_slice = voxels[scene, :, :, z_dim]

    plt.title('Scene: {}, z_dim: {}'.format(scene, z_dim))
    axarr[z_dim/4, z_dim%4].imshow(d_slice, interpolation='nearest')
  plt.show()
    
