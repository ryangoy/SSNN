import numpy as np
from pyntcloud.io import read_ply 
import sys
import pandas as pd


def ply2txt(ply_file, write_path):
  input_pc = read_ply(ply_file)
  input_pc = input_pc["points"].as_matrix(columns=["x", "y", "z", "r", "g", "b"])
  np.savetxt(write_path, input_pc)

if __name__ == '__main__':
  if len(sys.argv) < 2:
    print("Usage:\n\tpython ply2txt.py <path to ply file> <path to save file>\n")
    exit()
  ply2txt(sys.argv[1], sys.argv[2])