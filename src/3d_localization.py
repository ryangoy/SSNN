####################################################
# Main script to run. Loads data and interfaces
# with the SSNN object. 
#                        
# @author Ryan Goy
####################################################

import tensorflow as tf
import numpy as np
import os
from os.path import join, isdir
from os import listdir
from utils import normalize_pointclouds, load_points, create_jaccard_labels, save_output
from SSNN import SSNN
import time
from object_boundaries import generate_bounding_boxes

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Tensorflow flags boilerplate code.
flags = tf.app.flags
FLAGS = flags.FLAGS

# Define user inputs.
flags.DEFINE_string('data_dir', '/home/ryan/cs/datasets/SSNN/test', 
                    'Path to base directory.')
flags.DEFINE_bool('load_from_npy', True, 'Whether to load from preloaded \
                    dataset')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs to train.')
flags.DEFINE_float('val_split', 0.1, 'Percentage of input data to use as test.')
flags.DEFINE_integer('num_steps', 16, 'Number of intervals to sample\
                      from in each xyz direction.')
flags.DEFINE_integer('num_kernels', 4, 'Number of kernels to probe with.')
flags.DEFINE_integer('probes_per_kernel', 512, 'Number of sample points each\
                      kernel has.')

# Define constant paths
X_NPY         = join(FLAGS.data_dir, 'input_data.npy')
YS_NPY        = join(FLAGS.data_dir, 'segmentation_data.npy')
YL_NPY        = join(FLAGS.data_dir, 'label_data.npy')
OUTPUT_PATH   = join(FLAGS.data_dir, 'predictions.npy')
BBOX_PATH     = join(FLAGS.data_dir, 'bboxes.npy')

def main(_):
  
  # TODO: Preprocess input.
  # - remove outliers
  # - align to nearest 90 degree angle
  # - remove walls?
  # - data augmentation

  X_raw, ys_raw, yl = load_points(path=FLAGS.data_dir, X_npy_path=X_NPY,
                                  ys_npy_path = YS_NPY, yl_npy_path = YL_NPY, 
                                  load_from_npy=FLAGS.load_from_npy)

  print("Loaded {} pointclouds.".format(len(X_raw)))
  
  # Shift to the same coordinate space between pointclouds while getting the max
  # width, height, and depth dims of all rooms.
  print("Normalizing pointclouds...")
  X_cont, dims, ys = normalize_pointclouds(X_raw, ys_raw)
  dims = np.array([7.5, 7.5, 7.5])
  kernel_size = dims / FLAGS.num_steps
  print("Generating bboxes...")
  bboxes = generate_bounding_boxes(ys, BBOX_PATH)
  
  print("Processing labels...")
  y_cls, y_loc = create_jaccard_labels(bboxes, FLAGS.num_steps, kernel_size)

  np.save('cls_labels.npy', y_cls)
  np.save('loc_labels.npy', y_loc)

  X_ = []
  for sc in X_cont:
    X_.append([sc[0]])

  X_cont = np.array(X_)

  # Initialize model. max_room_dims and step_size are in meters.
  ssnn = SSNN(dims, num_kernels=FLAGS.num_kernels, 
                    probes_per_kernel=FLAGS.probes_per_kernel, 
                    probe_steps=FLAGS.num_steps)

  # Probe processing.
  print("Running probe operation...")
  probe_start = time.time()
  X = ssnn.probe(X_cont)
  probe_time = time.time() - probe_start
  print("Probe operation took {:.4f} seconds to run.".format(probe_time))

  X = np.squeeze(X, axis=1)

  # Used for developing so redudant calculations are omitted.
  # np.save('X.npy', X)
  # X = np.load('X.npy')

  p_mean = X.mean(axis=(4,5))

  np.save('probe_output.npy', p_mean)

  # Train model.
  train_split = int((FLAGS.val_split) * X.shape[0])
  X_trn = X[train_split:]
  y_trn_cls = y_cls[train_split:]
  y_trn_loc = y_loc[train_split:]
  X_val = X[:train_split]
  y_val_cls = y_cls[:train_split]
  y_val_loc = y_loc[:train_split]

  ssnn.train_val(X_trn, y_trn_cls, y_trn_loc, epochs=FLAGS.num_epochs) #y_l not used yet for localization

  # Test model. Using validation since we won't be using real 
  # "test" data yet. Preds will be an array of bounding boxes. 
  preds = ssnn.test(X_val)
  print preds.mean()

  # Save output.
  save_output('predictions.npy', preds)

# Tensorflow boilerplate code.
if __name__ == '__main__':
  tf.app.run()
