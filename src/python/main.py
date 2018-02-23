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
from utils import *
from SSNN import SSNN
import time
from object_boundaries import generate_bounding_boxes
import os
import psutil
from compute_bbox_accuracy import compute_accuracy


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Tensorflow flags boilerplate code.
flags = tf.app.flags
FLAGS = flags.FLAGS

# Define user inputs.
flags.DEFINE_string('data_dir', '/home/ryan/cs/datasets/SSNN/matterport/v1/scans', 
                    'Path to base directory.')
flags.DEFINE_bool('load_from_npy', True, 'Whether to load from preloaded \
                    dataset')
flags.DEFINE_integer('num_epochs', 200, 'Number of epochs to train.')
flags.DEFINE_float('val_split', 0.1, 'Percentage of input data to use as test.')
flags.DEFINE_float('learning_rate', 0.00001, 'Learning rate for training.')
flags.DEFINE_integer('num_steps', 64, 'Number of intervals to sample\
                      from in each xyz direction.')
flags.DEFINE_integer('k_size_factor', 3, 'Size of the probing kernel with respect to the step size.')
flags.DEFINE_integer('batch_size', 4, 'Batch size for training.')
flags.DEFINE_integer('num_kernels', 4, 'Number of kernels to probe with.')
flags.DEFINE_integer('probes_per_kernel', 16, 'Number of sample points each\
                      kernel has.')
flags.DEFINE_integer('num_dot_layers', 8, 'Number of dot product layers per kernel')
flags.DEFINE_float('loc_loss_lambda', 3, 'Relative weight of localization params.')
flags.DEFINE_integer('jittered_copies', 1, 'Number of times the dataset is copied and jittered for data augmentation.')

flags.DEFINE_string('checkpoint_save_dir', None, 'Path to saving checkpoint.')
flags.DEFINE_bool('checkpoint_load_dir', None, 'Path to loading checkpoint.')
flags.DEFINE_bool('load_probe_output', True, 'Load the probe output if a valid file exists.')



# DO NOT CHANGE
NUM_SCALES = 3
NUM_HOOK_STEPS = int(FLAGS.num_steps / 4)

# Define sets for training and testing
TRAIN_AREAS = ['Area_1', 'Area_2', 'Area_3', 'Area_4', 'Area_5']
TEST_AREAS = ['Area_6']

# Define constant paths
intermediate_dir = join(FLAGS.data_dir, 'intermediates')
if not exists(intermediate_dir):
  makedirs(intermediate_dir)
output_dir = join(FLAGS.data_dir, 'outputs')
if not exists(output_dir):
  makedirs(output_dir)

# raw inputs
X_TRN            = join(intermediate_dir, 'trn_data.npy')
YS_TRN           = join(intermediate_dir, 'trn_seg_labels.npy')
YL_TRN           = join(intermediate_dir, 'trn_cls_labels.npy')
PROBE_TRN        = join(intermediate_dir, 'trn_probe_out.npy')

X_TEST           = join(intermediate_dir, 'test_data.npy')
YS_TEST          = join(intermediate_dir, 'test_seg_labels.npy')
YL_TEST          = join(intermediate_dir, 'test_cls_labels.npy')
PROBE_TEST       = join(intermediate_dir, 'test_probe_out.npy')

# processed inputs and ouputs
CLS_TRN_LABELS   = join(output_dir, 'cls_trn_labels.npy')
LOC_TRN_LABELS   = join(output_dir, 'loc_trn_labels.npy')
BBOX_TRN_LABELS  = join(output_dir, 'bbox_trn_labels.npy')

CLS_TEST_LABELS  = join(output_dir, 'cls_test_labels.npy')
LOC_TEST_LABELS  = join(output_dir, 'loc_test_labels.npy')
BBOX_TEST_LABELS = join(output_dir, 'bbox_test_labels.npy')

CLS_PREDS        = join(output_dir, 'cls_predictions.npy')
LOC_PREDS        = join(output_dir, 'loc_predictions.npy')
BBOX_PREDS       = join(output_dir, 'bbox_predictions.npy')
BBOX_CLS_PREDS   = join(output_dir, 'bbox_cls_predictions.npy')


def preprocess_input(model, data_dir, areas, x_path, ys_path, yl_path, probe_path, 
                      cls_labels, loc_labels, bbox_labels, load_from_npy, load_probe_output, num_copies=0, is_train=True):

  # yl not used for now
  X_raw, yb_raw, yl, new_ds = load_points_matterport(path=data_dir, X_npy_path=x_path,
                                  yb_npy_path = ys_path, yl_npy_path = yl_path, 
                                  load_from_npy=load_from_npy, is_train=is_train)

  print("Loaded {} pointclouds.".format(len(X_raw)))
  process = psutil.Process(os.getpid())
  # Shift to the same coordinate space between pointclouds while getting the max
  # width, height, and depth dims of all rooms.

  print("Normalizing pointclouds...")
  X_cont, dims, ys = normalize_pointclouds_matterport(X_raw, yb_raw)
  # print("Augmenting dataset...")

  # X_cont, ys = augment_pointclouds(X_cont, ys, copies=num_copies)

  dims = np.array([7.5, 7.5, 7.5])
  kernel_size = dims / NUM_HOOK_STEPS
  # print("Generating bboxes...")
  # bboxes = generate_bounding_boxes(ys, bbox_labels)
  np.save(bbox_labels, ys)
  bboxes = ys

  print("Processing labels...")
  y_cls, y_loc = create_jaccard_labels(bboxes, NUM_HOOK_STEPS, kernel_size)
  np.save(cls_labels, y_cls)
  np.save(loc_labels, y_loc)

  # Hack-y way of combining samples into one array since each sample has a
  # different number of points.
  print("Combining samples...")

  # Probe processing.
  if exists(probe_path) and load_probe_output and not new_ds:
    # Used for developing so redudant calculations are omitted.
    print ("Loading previous probe output...")
    X = np.load(probe_path)
  else:
    print("Amount of memory used before probing: {}GB".format(process.memory_info().rss // 1e9))
    print("Running probe operation...")
    probe_start = time.time()


    probe_shape = (len(X_cont), NUM_HOOK_STEPS, NUM_HOOK_STEPS, NUM_HOOK_STEPS, FLAGS.num_kernels, FLAGS.probes_per_kernel)
    X, problem_pcs = model.probe(X_cont, probe_shape, probe_path)
    probe_time = time.time() - probe_start
    print("Probe operation took {:.4f} seconds to run.".format(probe_time))
    
    print("Amount of memory used after probing: {}GB".format(process.memory_info().rss // 1e9))
    

    for problem_pc in problem_pcs:
      y_cls[problem_pc] = y_cls[problem_pc-1]
      y_loc[problem_pc] = y_loc[problem_pc-1]



  print("Finished pre-processing.")
  return X, y_cls, y_loc



def main(_):
  dims = np.array([7.5, 7.5, 7.5])
  kernel_size = dims / FLAGS.num_steps
  # Initialize model. max_room_dims and step_size are in meters.
  ssnn = SSNN(dims, num_kernels=FLAGS.num_kernels, 
                    probes_per_kernel=FLAGS.probes_per_kernel, 
                    probe_steps=FLAGS.num_steps, probe_hook_steps=NUM_HOOK_STEPS,
                    num_scales=NUM_SCALES,
                    dot_layers=FLAGS.num_dot_layers,
                    ckpt_save=FLAGS.checkpoint_save_dir,
                    loc_loss_lambda=FLAGS.loc_loss_lambda,
                    learning_rate=FLAGS.learning_rate,
                    k_size_factor=FLAGS.k_size_factor)


  load_probe = FLAGS.load_probe_output and FLAGS.load_from_npy
  X, y_cls, y_loc = preprocess_input(ssnn, FLAGS.data_dir, TRAIN_AREAS, X_TRN, YS_TRN, YL_TRN, PROBE_TRN, 
                      CLS_TRN_LABELS, LOC_TRN_LABELS, BBOX_TRN_LABELS, FLAGS.load_from_npy,
                      load_probe, num_copies=FLAGS.jittered_copies)


  X_test, _, _ = preprocess_input(ssnn, FLAGS.data_dir, TEST_AREAS, X_TEST, YS_TEST, YL_TEST, PROBE_TEST, 
                      CLS_TEST_LABELS, LOC_TEST_LABELS, BBOX_TEST_LABELS, FLAGS.load_from_npy,
                      load_probe, is_train=False)


  # Train model.
  train_split = int((FLAGS.val_split) * X.shape[0])
  X_trn = X[train_split:]
  y_trn_cls = y_cls[train_split:]
  y_trn_loc = y_loc[train_split:]
  X_val = X[:train_split]
  y_val_cls = y_cls[:train_split]
  y_val_loc = y_loc[:train_split]
  print("Beginning training...")

  ssnn.train_val(X_trn, y_trn_cls, y_trn_loc, X_val, y_val_cls, y_val_loc, epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size) #y_l not used yet for localization

  # Test model. Using validation since we won't be using real 
  # "test" data yet. Preds will be an array of bounding boxes. 
  start_test = time.time()
  cls_preds, loc_preds = ssnn.test(X_test)
  end_test = time.time()

  print("Time to run {} test samples took {} seconds.".format(X_test.shape[0], end_test-start_test))
  
  # Save output.
  save_output(CLS_PREDS, LOC_PREDS, cls_preds, loc_preds, 
                             NUM_HOOK_STEPS, NUM_SCALES)
  

  cls_f = np.load(CLS_PREDS)
  loc_f = np.load(LOC_PREDS)


  bboxes = output_to_bboxes(cls_f, loc_f, NUM_HOOK_STEPS, NUM_SCALES, 
                            dims/NUM_HOOK_STEPS, BBOX_PREDS, BBOX_CLS_PREDS)

  compute_accuracy(BBOX_PREDS, BBOX_TEST_LABELS)
  
# Tensorflow boilerplate code.
if __name__ == '__main__':
  tf.app.run()
