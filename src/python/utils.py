import numpy as np
import sys
import paths
from os.path import join, exists
import time
import knn



def load_points(path=None, npy_path=None):
  if npy_path is not None and exists(npy_path):
    xyzrgb = np.load(npy_path)
  else:
    xyzrgb = np.loadtxt(path)
  return xyzrgb

print dir(knn)

