####################################################
# Main script to run. Loads data and interfaces
# with the SSNN object. 
#                        
# @author Ryan Goy
####################################################

import tensorflow as tf
import numpy as np
import os
from os.path import join, isdir, exists
from os import listdir, makedirs
from utils import normalize_pointclouds, load_points, create_jaccard_labels, save_output, output_to_bboxes, nms
from SSNN import SSNN
import time
from object_boundaries import generate_bounding_boxes

os.environ["CUDA_VISIBLE_DEVICES"] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Tensorflow flags boilerplate code.
flags = tf.app.flags
FLAGS = flags.FLAGS

# Define user inputs.
flags.DEFINE_string('data_dir', '/home/ryan/cs/datasets/SSNN/buildings', 
                    'Path to base directory.')
flags.DEFINE_bool('load_from_npy', True, 'Whether to load from preloaded \
                    dataset')
flags.DEFINE_integer('num_epochs', 20, 'Number of epochs to train.')
flags.DEFINE_float('val_split', 0.1, 'Percentage of input data to use as test.')
flags.DEFINE_integer('num_steps', 16, 'Number of intervals to sample\
                      from in each xyz direction.')
flags.DEFINE_integer('num_kernels', 32, 'Number of kernels to probe with.')
flags.DEFINE_integer('probes_per_kernel', 256, 'Number of sample points each\
                      kernel has.')
flags.DEFINE_string('checkpoint_save_dir', None, 'Path to saving checkpoint.')
flags.DEFINE_bool('checkpoint_load_dir', None, 'Path to loading checkpoint.')
flags.DEFINE_bool('load_probe_output', True, 'Load the probe output if a valid file exists.')


# DO NOT CHANGE
NUM_SCALES = 3

# Define constant paths
intermediate_dir = join(FLAGS.data_dir, 'intermediates')
if not exists(intermediate_dir):
  makedirs(intermediate_dir)
X_NPY         = join(intermediate_dir, 'input_data.npy')
YS_NPY        = join(intermediate_dir, 'segmentation_data.npy')
YL_NPY        = join(intermediate_dir, 'label_data.npy')
OUTPUT_PATH   = join(intermediate_dir, 'predictions.npy')
BBOX_PATH     = join(intermediate_dir, 'bboxes.npy')
PROBE_NPY     = join(intermediate_dir, 'probe_out.npy')
CLS_LABELS    = 'cls_labels.npy'
LOC_LABELS    = 'loc_labels.npy'
BBOX_LABELS   = 'bbox_labels.npy'
CLS_PREDS     = 'cls_predictions.npy'
LOC_PREDS     = 'loc_predictions.npy'
NMS_PREDS     = 'nms_loc_predictions.npy'
BBOX_PREDS    = 'bbox_predictions.npy'
BBOX_CLS_PREDS= 'bbox_cls_predictions.npy'

def main(_):
  
  # TODO: Preprocess input.
  # - remove outliers
  # - align to nearest 90 degree angle
  # - data augmentation
  X_raw, ys_raw, yl, new_ds = load_points(path=FLAGS.data_dir, X_npy_path=X_NPY,
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
  np.save(BBOX_LABELS, bboxes)
  print("Processing labels...")
  y_cls, y_loc = create_jaccard_labels(bboxes, FLAGS.num_steps, kernel_size)
  np.save(CLS_LABELS, y_cls)
  np.save(LOC_LABELS, y_loc)

  # Hack-y way of combining samples into one array since each sample has a
  # different number of points.
  X_ = []
  for sc in X_cont:
    X_.append([sc[0]])
  X_cont = np.array(X_)

  # Initialize model. max_room_dims and step_size are in meters.
  ssnn = SSNN(dims, num_kernels=FLAGS.num_kernels, 
                    probes_per_kernel=FLAGS.probes_per_kernel, 
                    probe_steps=FLAGS.num_steps, num_scales=NUM_SCALES,
                    ckpt_save=FLAGS.checkpoint_save_dir)

  # Probe processing.
  if exists(PROBE_NPY) and FLAGS.load_probe_output and not new_ds:
    # Used for developing so redudant calculations are omitted.
    print ("Loading previous probe output...")
    X = np.load(PROBE_NPY)
  else:
    print("Running probe operation...")
    probe_start = time.time()
    X = ssnn.probe(X_cont)
    probe_time = time.time() - probe_start
    print("Probe operation took {:.4f} seconds to run.".format(probe_time))
    X = np.squeeze(X, axis=1)
    np.save(PROBE_NPY, X)

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
  cls_preds, loc_preds = ssnn.test(X_val)
  
  # Save output.
  cls_f, loc_f = save_output(CLS_PREDS, LOC_PREDS, NMS_PREDS, cls_preds, loc_preds, 
                             FLAGS.num_steps, NUM_SCALES)
  bboxes = output_to_bboxes(cls_f, loc_f, FLAGS.num_steps, NUM_SCALES, 
                            kernel_size, BBOX_PREDS, BBOX_CLS_PREDS)

# Tensorflow boilerplate code.
if __name__ == '__main__':
  tf.app.run()
