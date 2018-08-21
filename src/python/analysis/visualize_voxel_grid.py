# visualizes the output of the dot product layer or a pooling layer

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import sys

# If visualizing the output of a pooling layer
vis_pool=True 

grid = np.load(sys.argv[1])
grid = grid+grid.min()

print('loaded successfully')

grid = np.transpose(grid, (0,2,3,4,1))

# loop through each scene
for i in range(grid.shape[0]):
	# prepare some coordinates
	


	if vis_pool:
		# just visualize the first filter response
		weightings = grid[i, :,:,:, 0]
	else:
		weightings = np.sum(grid[i, :,:,:], axis=(-1, -2, -3))

	print(weightings.shape)
	x, y, z = np.indices(weightings.shape)

	# and plot everything
	fig = plt.figure()
	# ax = fig.gca(projection='3d')
	# ax.scatter(x, y, z, c=weightings.flatten(), cmap="Blues", alpha=0.2)
	# plt.show()
	plt.imshow(np.max(weightings, -1))

	plt.show()
