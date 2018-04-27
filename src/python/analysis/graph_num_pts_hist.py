# Graphs a histogram of the number of points in a pointcloud (input is .npy file)

import numpy as np
import sys
import matplotlib.pyplot as plt

# shape: (num_pcs, num_pts, num_channels)
pcs = np.load(sys.argv[1])
num_pts = []

for pc in pcs:
  num_pts.append(pc.shape[0])
plt.hist(num_pts, bins=20, range=(0, 2e6))
plt.show()
