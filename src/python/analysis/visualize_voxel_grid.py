import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


vis_pool=True

if vis_pool:
	grid = np.load('/home/ryan/Desktop/pool1.npy')
else:
	grid = np.load('/home/ryan/Desktop/pc_batch.npy')


for i in range(grid.shape[0]):
	# prepare some coordinates
	x, y, z = np.indices(grid.shape[1:4])

	if vis_pool:
		weightings = grid[i, :,:,:, 1]
	else:
		weightings = np.sum(grid[i, :,:,:], axis=(-1, -2, -3))

	# and plot everything
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# ax.voxels(grid[i,:,:,:,0],  edgecolor='k')
	ax.scatter(x, y, z, c=weightings.flatten(), cmap="Blues", alpha=0.2)
	plt.show()
	plt.imshow(np.max(weightings, -1))

	plt.show()
