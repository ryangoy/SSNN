####################################################
# Main script to run. Loads data and interfaces
# with the SSNN object. 
#                        
# @author Ryan Goy
####################################################

import tensorflow as tf
import numpy as np
from os.path import join, isdir
from os import listdir
from utils import get_dims, normalize_pointclouds, load_points
from SSNN import SSNN
import time
from object_boundaries import generate_bounding_boxes


# Tensorflow flags boilerplate code.
flags = tf.app.flags
FLAGS = flags.FLAGS

# Define user inputs.
flags.DEFINE_string('data_dir', '/home/ryan/cs/datasets/SSNN/test', 
                    'Path to base directory.')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to train.')
flags.DEFINE_float('val_split', 0.1, 'Percentage of input data to use as test.')
flags.DEFINE_integer('num_probe_steps', 10, 'Number of intervals to sample\
                      from in each xyz direction.')

# Define constant paths
X_NPY         = join(FLAGS.data_dir, 'input_data.npy')
YS_NPY        = join(FLAGS.data_dir, 'segmentation_data.npy')
YL_NPY        = join(FLAGS.data_dir, 'label_data.npy')
OUTPUT_PATH   = join(FLAGS.data_dir, 'predictions.npy')

def main(_):

  X_raw, ys, yl = load_points(path=FLAGS.data_dir, X_npy_path=X_NPY,
    ys_npy_path = YS_NPY, yl_npy_path = YL_NPY, load_from_npy=True)

  # Shift to the same coordinate space between pointclouds while getting the max
  # width, height, and depth dims of all rooms.
  X_cont, dims = normalize_pointclouds(X_raw)
  bboxes = generate_bounding_boxes(ys)

  # TODO: Preprocess input.
  # - remove outliers
  # - align to nearest 90 degree angle
  # - remove walls?
  # - data augmentation

  # Initialize model. max_room_dims and step_size are in meters.
  ssnn = SSNN(dims, num_kernels=1, probes_per_kernel=1, probe_steps=10)

  # Probe processing.
  probe_start = time.time()
  X = ssnn.probe(X_cont)
  probe_time = time.time() - probe_start
  print("Probe operation took {:.4f} seconds to run.".format(probe_time))

  # Train model.
  train_split = (1-FLAGS.val_split) * X.shape[0]
  X_trn = X[:train_split]
  y_trn = bboxes[:train_split]
  X_val = X[train_split:]
  y_val = bboxes[train_split:]
  ssnn.train_val(X_trn, y_trn) #y_l not used yet for localization

  # Test model. Using validation since we won't be using real 
  # "test" data yet. Preds will be an array of bounding boxes. 
  preds = ssnn.test(X_val)

  # Save output.
  np.save(OUTPUT_PATH, preds)

# Tensorflow boilerplate code.
if __name__ == '__main__':
  tf.app.run()
