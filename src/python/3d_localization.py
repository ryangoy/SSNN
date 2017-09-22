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

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/home/ryan/cs/datasets/SSNN/test', 'Path to base directory.')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to train.')
flags.DEFINE_float('val_split', 0.2, 'Percentage of input data to use as test.')

X_NPY         = join(FLAGS.data_dir, 'input_data.npy')
YS_NPY        = join(FLAGS.data_dir, 'segmentation_data.npy')
YL_NPY        = join(FLAGS.data_dir, 'label_data.npy')
OUTPUT_PATH   = join(FLAGS.data_dir, 'predictions.npy')

def load_from_directory(path):
  """
  Assumes dataset structure is as follows:
  base
    area_1
      room_1
      Annotations
      room_1.txt
      ...
      room_n
    ...
    area_n
  """
  input_data = []
  segmentations = []
  labels = []
  # Loop through Areas
  for area in listdir(path):
    print "Loading area {}...".format(area)
    area_path = join(path, area)
    if not isdir(area_path):
      continue
      
    # Loop through rooms
    for room in listdir(area_path):
      print "\tLoading room {}...".format(room)
      room_path = join(area_path, room)
      if not isdir(room_path) or room == '.DS_Store' or room.endswith('Angle.txt'):
        continue
        
      # Load point cloud
      input_pc = np.loadtxt(join(room_path, room+'.txt'), dtype=np.float32)
      
      # Loop and load Annotations folder
      annotation_pc = []
      annotation_label = []
      for annotation in listdir(join(room_path, 'Annotations')):
        if annotation.startswith('wall') or annotation.startswith('ceiling') or not annotation.endswith('.txt'):
          continue
        annotation_pc.append(np.loadtxt(join(room_path, 'Annotations', annotation), dtype=np.float32))
        annotation_label.append(annotation.split('.')[0])
      annotation_pc = np.array(annotation_pc)
      
      input_data.append(input_pc)
      segmentations.append(annotation_pc)
      labels.append(annotation_label)

  input_data = np.array(input_data)
  segmentations = np.array(segmentations)
  labels = np.array(labels)

  # print input_data.shape
  # print segmentations.shape
  # print labels.shape

  return input_data, segmentations, labels

def load_from_npy(X_path, ys_path, yl_path):
  return np.load(X_path), np.load(ys_path), np.load(yl_path)

def main(_):

  # Load data from preloaded npy files or from directory.
  if FLAGS.load_from_npy:
    X, ys, yl = load_from_npy(X_NPY, YS_NPY, YL_NPY)
  else:
    X, ys, yl = load_from_directory(FLAGS.data_dir)
    np.save(X_NPY_PATH, X)
    np.save(YS_NPY_PATH, ys)
    np.save(YL_NPY_PATH, yl)

  # Preprocess input.
  ### TO DO ###
  # -remove outliers
  # -align to nearest 90 degree angle
  # -remove walls?
  # -data augmentation

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

if __name__ == '__main__':
  tf.app.run()
