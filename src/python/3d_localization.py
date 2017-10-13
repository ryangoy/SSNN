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

# Tensorflow flags boilerplate code.
flags = tf.app.flags
FLAGS = flags.FLAGS

# Define user inputs.
flags.DEFINE_string('data_dir', '/home/ryan/cs/datasets/SSNN/test', 
                    'Path to base directory.')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to train.')
flags.DEFINE_float('val_split', 0.1, 'Percentage of input data to use as test.')

# Define constant paths
X_NPY         = join(FLAGS.data_dir, 'input_data.npy')
YS_NPY        = join(FLAGS.data_dir, 'segmentation_data.npy')
YL_NPY        = join(FLAGS.data_dir, 'label_data.npy')
OUTPUT_PATH   = join(FLAGS.data_dir, 'predictions.npy')

def main(_):

  X_raw, ys, yl = load_points(path=None, npy_path=None, load_from_npy=True)

  # Get the dimensions of the first room. 
  room_dims = get_dims(X_raw[0])

  # Shift to the same coordinate space between pointclouds.
  X = normalize_pointclouds(X_raw)

  # TODO: Preprocess input.
  # - remove outliers
  # - align to nearest 90 degree angle
  # - remove walls?
  # - data augmentation

  # Initialize model. max_room_dims and step_size are in meters.
  ssnn = SSNN(max_room_dims, step_size)

  # Train model.
  train_split = (1-FLAGS.val_split) * X.shape[0]
  X_trn = X[:train_split]
  ys_trn = ys[:train_split]
  X_val = X[train_split:]
  ys_val = ys[train_split:]
  ssnn.train_val(X_trn, ys_trn) #y_l not used yet for localization

  # Test model. Using validation since we won't be using real 
  # "test" data yet. Preds will be an array of bounding boxes. 
  preds = ssnn.test(X_val)

  # Save output.
  np.save(OUTPUT_PATH, preds)

# Tensorflow boilerplate code.
if __name__ == '__main__':
  tf.app.run()
