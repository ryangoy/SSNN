import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


vis_pool=False

if vis_pool:
	grid = np.load('pool1.npy')
else:
	grid = np.load('pc_batch.npy')


for i in range(grid.shape[0]):
	# prepare some coordinates
	x, y, z = np.indices(grid.shape[1:4])

	# # draw cuboids in the top left and bottom right corners, and a link between them
	# cube1 = (x < 3) & (y < 3) & (z < 3)
	# print cube1
	# cube2 = (x >= 5) & (y >= 5) & (z >= 5)
	# link = abs(x - y) + abs(y - z) + abs(z - x) <= 2

	# # combine the objects into a single boolean array
	# voxels = cube1 | cube2 | link

	# # set the colors of each object
	# colors = np.empty(voxels.shape, dtype=object)
	# colors[link] = 'red'
	# colors[cube1] = 'blue'
	# colors[cube2] = 'green'

	if vis_pool:
		weightings = grid[i, :,:,:, 0]
	else:
		weightings = np.max(grid[i, :,:,:], axis=(-1, -2, -3))

	# and plot everything
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# ax.voxels(grid[i,:,:,:,0],  edgecolor='k')
	ax.scatter(x, y, z, c=weightings.flatten(), cmap="Blues", alpha=0.2)
	plt.show()
	plt.imshow(np.max(weightings, -1))

	plt.show()