import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

pcs = np.load('pc.npy')


for i in range(pcs.shape[0]):
	ss_factor = 1000

	x = pcs[i, ::ss_factor, ::ss_factor, ::ss_factor, 0, 0, 0]
	y = pcs[i, ::ss_factor, ::ss_factor, ::ss_factor, 0, 0, 0]
	z = pcs[i, ::ss_factor, ::ss_factor, ::ss_factor, 0, 0, 0]

	# and plot everything
	fig = plt.figure()
	ax = fig.gca(projection='3d')
	# ax.voxels(grid[i,:,:,:,0],  edgecolor='k')
	ax.scatter(x, y, z, cmap="Blues", alpha=0.2)

	plt.show()