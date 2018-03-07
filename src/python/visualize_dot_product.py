import numpy as np
import sys
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


def visualize_dot_product(probe_coords, dot_product_weights, kernel_id=0):
	kernel = probe_coords[kernel_id]
	for dp_id in range(dot_product_weights.shape[-1]):
		xs = probe_coords[kernel_id, :, 0]
		ys = probe_coords[kernel_id, :, 1]
		zs = probe_coords[kernel_id, :, 2]

		weightings = dot_product_weights[kernel_id, :, dp_id]
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')

		intensities = np.array([weightings, weightings, weightings]).T

		ax.scatter(xs, ys, zs, c=weightings, cmap="Blues")

		ax.set_xlabel('X')
		ax.set_ylabel('Y')
		ax.set_zlabel('Z')

		plt.show()


if __name__ == '__main__':
	# This should be shape (num_kernels, num_probes, 3)
	probe_coords = np.load(sys.argv[1])
	# This should be (num_kernels, num_probes, num_dot_layers)
	dot_product_weights = np.load(sys.argv[2])
	visualize_dot_product(probe_coords, dot_product_weights)