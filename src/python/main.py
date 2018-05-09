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
from compute_mAP3 import compute_mAP
import pickle as pkl
from matplotlib.colors import rgb_to_hsv

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# Tensorflow flags boilerplate code.
flags = tf.app.flags
FLAGS = flags.FLAGS

#########
# FLAGS #
#########

# Data information: loading and saving options.
flags.DEFINE_string('data_dir', '/home/ryan/cs/datasets/SSNN/buildings', 'Path to base directory.')
flags.DEFINE_string('dataset_name', 'stanford', 'Name of dataset. Supported datasets are [stanford, matterport].')
flags.DEFINE_bool('load_from_npy', False, 'Whether to load from preloaded dataset')
flags.DEFINE_bool('load_probe_output', False, 'Load the probe output if a valid file exists.')
flags.DEFINE_integer('rotated_copies', 3, 'Number of times the dataset is copied and rotated for data augmentation.')
flags.DEFINE_string('checkpoint_save_dir', None, 'Path to saving checkpoint.')
flags.DEFINE_string('checkpoint_load_dir', None, 'Path to loading checkpoint.')
flags.DEFINE_integer('checkpoint_load_iter', 50, 'Iteration from save dir to load.')
flags.DEFINE_float('checkpoint_save_interval', 10, 'If checkpoint_save_interval is defined, then sets save interval.')
flags.DEFINE_boolean('use_rgb', True, 'If True, then loads colored pointclouds. Else, loads uncolored pointclouds.')
flags.DEFINE_string('single_class', None, 'Class name for single object detector.')
flags.DEFINE_boolean('train', True, 'If True, the model trains and validates.')
flags.DEFINE_boolean('test', True, 'If True, the model tests as long as it load from a valid checkpoint or follow after training.')

# Training hyperparameters.
flags.DEFINE_integer('num_epochs', 100, 'Number of epochs to train.')
flags.DEFINE_float('test_split', 0.1, 'Percentage of input data to use as test data.')
flags.DEFINE_float('val_split', 0.1, 'Percentage of input data to use as validation. Taken after the test split.')
flags.DEFINE_float('learning_rate', 0.00005, 'Learning rate for training.')
flags.DEFINE_float('loc_loss_lambda', 3, 'Relative weight of localization params.')
flags.DEFINE_float('dropout', 0.5, 'Keep probability for layers with dropout.')

# Probing hyperparameters.
flags.DEFINE_integer('num_xy_steps', 128, 'Number of intervals to sample from in each xyz direction.')
flags.DEFINE_integer('num_z_steps', 16, 'Number of intervals to sample from in each xyz direction.')
flags.DEFINE_integer('k_size_factor', 3, 'Size of the probing kernel with respect to the step size.')
flags.DEFINE_integer('batch_size', 4, 'Batch size for training.')
flags.DEFINE_integer('num_kernels', 4, 'Number of kernels to probe with.')
flags.DEFINE_integer('probes_per_kernel', 64, 'Number of sample points each kernel has.')
flags.DEFINE_integer('num_dot_layers', 16, 'Number of dot product layers per kernel')
flags.DEFINE_integer('num_anchors', 4, 'Number of anchors to use.')

# DO NOT CHANGE
NUM_SCALES = 3
NUM_HOOK_STEPS = int(FLAGS.num_z_steps)
DIMS = np.array([FLAGS.num_xy_steps, FLAGS.num_xy_steps, FLAGS.num_z_steps])

# Define sets for training and testing (Stanford dataset)
TRAIN_AREAS = ['Area_6', 'Area_2', 'Area_3', 'Area_4', 'Area_5'] 
TEST_AREAS = ['Area_1']

# Define categories.
# CATEGORIES = ['box', 'picture', 'pillow', 'curtain', 'table', 'bench', 'side table', 'window', 'bed', 'tv', 
#                   'heater', 'pot', 'bottles', 'washbasin', 'light', 'clothes', 'bin', 'cabinet', 'radiator', 'bookcase',
#                   'button', 'toilet paper', 'toilet', 'control panel', 'towel']

if FLAGS.single_class is None:
  if FLAGS.dataset_name == 'stanford':
    CATEGORIES = ['sofa', 'table', 'chair', 'board']
  else:
    CATEGORIES = ['bathtub', 'bed', 'bookshelf', 'chair', 'desk', 'dresser', 'nightstand', 'sofa', 'table', 'toilet']
    #CATEGORIES = ['sofa', 'table', 'chair', 'board']
else:
  CATEGORIES = [FLAGS.single_class]

# Define constant paths (TODO: make this more organized between datasets)
intermediate_dir = join(FLAGS.data_dir, 'intermediates')
if not exists(intermediate_dir):
  makedirs(intermediate_dir)
output_dir = join(FLAGS.data_dir, 'outputs')
if not exists(output_dir):
  makedirs(output_dir)

# Raw inputs
X_TRN            = join(intermediate_dir, 'trn_data.npy')
YS_TRN           = join(intermediate_dir, 'trn_seg_labels.npy')
YL_TRN           = join(intermediate_dir, 'trn_cls_labels.npy')
PROBE_TRN        = join(intermediate_dir, 'trn_probe_out.npy') # memmap

X_TEST           = join(intermediate_dir, 'test_data.npy')
YS_TEST          = join(intermediate_dir, 'test_seg_labels.npy')
YL_TEST          = join(intermediate_dir, 'test_cls_labels.npy')
PROBE_TEST       = join(intermediate_dir, 'test_probe_out.npy') # memmap

# Processed inputs and ouputs
CLS_TRN_LABELS   = join(output_dir, 'cls_trn_labels.npy')
LOC_TRN_LABELS   = join(output_dir, 'loc_trn_labels.npy')
BBOX_TRN_LABELS  = join(output_dir, 'bbox_trn_labels.npy')
CLS_TRN_BBOX     = join(output_dir, 'bbox_trn_cls_labels.npy')

CLS_TEST_LABELS  = join(output_dir, 'cls_test_labels.npy')
LOC_TEST_LABELS  = join(output_dir, 'loc_test_labels.npy')
BBOX_TEST_LABELS = join(output_dir, 'bbox_test_labels.npy')
CLS_TEST_BBOX    = join(output_dir, 'bbox_test_cls_labels.npy')

CLS_PREDS        = join(output_dir, 'cls_predictions.npy')
LOC_PREDS        = join(output_dir, 'loc_predictions.npy')
BBOX_PREDS       = join(output_dir, 'bbox_predictions.npy')
BBOX_CLS_PREDS   = join(output_dir, 'bbox_cls_predictions.npy')

MAPPING          = join(output_dir, 'mapping.pkl')


POSSIBLE_ANCHORS =  np.array([[1.0, 1.0, 1.0],
                              [2.0, 1.0, 1.0],
                              [1.0, 2.0, 1.0],
                              [2.0, 2.0, 1.0],
                              [0.5, 1.0, 1.0],
                              [1.0, 0.5, 1.0],
                              [0.5, 0.5, 1.0],
                              [1.0, 1.0, 2.0],
                              [1.0, 1.0, 0.5]])

ANCHORS = POSSIBLE_ANCHORS[:FLAGS.num_anchors]

# ANCHORS = np.array([[2.0, 2.0, 1.0]])

def preprocess_input(model, data_dir, areas, x_path, ys_path, yl_path, probe_path, 
                      cls_labels, loc_labels, bbox_labels, cls_by_box, load_from_npy, load_probe_output, num_copies=0, is_train=True, oh_mapping=None):
  """
  Converts raw data into form that can be fed into the ML pipeline. Operations include normalization, augmentation, 
  label ggeneration, and probing.
  """

  input_type = "train" if is_train else "test"
  assert FLAGS.dataset_name in ['stanford', 'matterport'], 'Supported datasets are stanford and matterport.'

  print("Running pre-processing for {} set.".format(input_type))
  if False and FLAGS.dataset_name == 'stanford':
    normalize_pointclouds_fn = normalize_pointclouds_stanford

  elif True or FLAGS.dataset_name == 'matterport':
    normalize_pointclouds_fn = normalize_pointclouds_matterport

  if FLAGS.dataset_name == 'matterport':
    X_raw, yb_raw, yl, new_ds = load_points_matterport(path=data_dir, X_npy_path=x_path,
                                    yb_npy_path = ys_path, yl_npy_path = yl_path, 
                                    load_from_npy=load_from_npy, is_train=is_train,
                                    categories=CATEGORIES, train_test_split=1.0 - FLAGS.test_split, use_rgb=FLAGS.use_rgb)
  elif FLAGS.dataset_name == 'stanford':
    X_raw, yb_raw, yl, new_ds = load_points_stanford(path=data_dir, X_npy_path=x_path,
                                  ys_npy_path = ys_path, yl_npy_path = yl_path, 
                                  load_from_npy=load_from_npy, areas=areas, categories=CATEGORIES)

  print("\tConverting RGB to HSV...")
  for X_rgb in X_raw:
    X_rgb[:, 3:] = rgb_to_hsv(X_rgb[:,3:])

  print("\tLoaded {} pointclouds for {}.".format(len(X_raw), input_type))
  process = psutil.Process(os.getpid())
 
  # Shift to the same coordinate space between pointclouds while getting the max
  # width, height, and depth dims of all rooms.


  if FLAGS.dataset_name == 'stanford':
    print("\tGenerating bboxes...")
    bboxes = generate_bounding_boxes(yb_raw, bbox_labels)
  elif FLAGS.dataset_name == 'matterport':
    bboxes = yb_raw
  


  print("\tAugmenting dataset...")
  X_raw, bboxes, yl = rotate_pointclouds(X_raw, bboxes, yl, num_rotations=num_copies)


  print("\tNormalizing pointclouds...")
  X_cont, dims, bboxes = normalize_pointclouds_fn(X_raw, bboxes, DIMS)

  np.save(bbox_labels, bboxes)

  yl = np.array(yl)
  kernel_size = DIMS / NUM_HOOK_STEPS



  print("\tProcessing labels...")
  y_cat_one_hot, mapping = one_hot_vectorize_categories(yl, mapping=oh_mapping)
  np.save(cls_by_box, y_cat_one_hot)
  y_cls, y_loc = create_jaccard_labels(bboxes, y_cat_one_hot, len(mapping)+1, NUM_HOOK_STEPS, kernel_size, ANCHORS)

  np.save(cls_labels, y_cls)
  np.save(loc_labels, y_loc)

  # Probe processing.
  if exists(probe_path) and load_probe_output and not new_ds:
    # Used for developing so redudant calculations are omitted.
    print ("\tLoading previous probe output...")
    X = np.memmap(probe_path, dtype='float32', mode='r', shape=(len(X_cont), FLAGS.num_x_steps, 
                             FLAGS.num_y_steps, FLAGS.num_z_steps, FLAGS.num_kernels, FLAGS.probes_per_kernel, 4))
  else:
    print("\tAmount of memory used before probing: {}GB".format(process.memory_info().rss // 1e9))
    print("\tRunning probe operation...")
    probe_start = time.time()
    probe_shape = (len(X_cont), NUM_HOOK_STEPS, NUM_HOOK_STEPS, NUM_HOOK_STEPS, FLAGS.num_kernels, FLAGS.probes_per_kernel)
    X, problem_pcs = model.probe(X_cont, probe_shape, probe_path)
    probe_time = time.time() - probe_start
    print("\tProbe operation took {:.4f} seconds to run.".format(probe_time))
    print("\tAmount of memory used after probing: {}GB".format(process.memory_info().rss // 1e9))
    
    # TODO: delete hard-coded elements of problem pointcloud removal (see SSNN.py counter var if/else logic).
    for problem_pc in problem_pcs:
      y_cls[problem_pc] = y_cls[problem_pc-1]
      y_loc[problem_pc] = y_loc[problem_pc-1]

  print("\tFinished pre-processing of {} set.".format(input_type))
  return X, y_cls, y_loc, y_cat_one_hot, bboxes, mapping

def main(_):


  # Initialize model. max_room_dims and step_size are in meters.
  ssnn = SSNN(DIMS, num_kernels=FLAGS.num_kernels, 
                    probes_per_kernel=FLAGS.probes_per_kernel, 
                    probe_xy_steps=FLAGS.num_xy_steps, 
                    probe_z_steps=FLAGS.num_z_steps,
                    probe_hook_steps=NUM_HOOK_STEPS,
                    num_scales=NUM_SCALES,
                    dot_layers=FLAGS.num_dot_layers,
                    ckpt_save=FLAGS.checkpoint_save_dir,
                    ckpt_load=FLAGS.checkpoint_load_dir,
                    ckpt_load_iter=FLAGS.checkpoint_load_iter,
                    loc_loss_lambda=FLAGS.loc_loss_lambda,
                    learning_rate=FLAGS.learning_rate,
                    dropout=FLAGS.dropout,
                    k_size_factor=FLAGS.k_size_factor,
                    num_classes=len(CATEGORIES)+1,
                    anchors=ANCHORS)


  load_probe = FLAGS.load_probe_output and FLAGS.load_from_npy

  if FLAGS.train:
    # Pre-process train data. Train/test data pre-processing is split for easier data streaming.
    X, y_cls, y_loc, y_cat_one_hot, bboxes, mapping = preprocess_input(ssnn, FLAGS.data_dir, TRAIN_AREAS, X_TRN, YS_TRN, YL_TRN, PROBE_TRN, 
                        CLS_TRN_LABELS, LOC_TRN_LABELS, BBOX_TRN_LABELS, CLS_TRN_BBOX, FLAGS.load_from_npy,
                        load_probe, num_copies=FLAGS.rotated_copies)

    # Train model.
    train_split = int((FLAGS.val_split) * X.shape[0])
    X_trn = X[train_split:]
    y_trn_cls = y_cls[train_split:]
    y_trn_loc = y_loc[train_split:]
    y_trn_one_hot = y_cat_one_hot[train_split:]
    trn_bboxes = bboxes[train_split:]
    np.save('y_cls.npy', y_trn_cls)
    X_val = X[:train_split]
    y_val_cls = y_cls[:train_split]
    y_val_loc = y_loc[:train_split]
    y_val_one_hot = y_cat_one_hot[:train_split]
    val_bboxes = bboxes[:train_split]
    print("Beginning training...")
    ssnn.train_val(X_trn, y_trn_cls, y_trn_loc, X_val, y_val_cls, y_val_loc, val_bboxes, y_val_one_hot, epochs=FLAGS.num_epochs, batch_size=FLAGS.batch_size, save_interval=FLAGS.checkpoint_save_interval)
    pkl.dump(mapping, open(MAPPING, 'wb'))

  if FLAGS.test:
    mapping=None
    #mapping = pkl.load(open(MAPPING, 'rb'))
    # Pre-process test data.
    X_test, _, _, _, _, _ = preprocess_input(ssnn, FLAGS.data_dir, TEST_AREAS, X_TEST, YS_TEST, YL_TEST, PROBE_TEST, 
                        CLS_TEST_LABELS, LOC_TEST_LABELS, BBOX_TEST_LABELS, CLS_TEST_BBOX, FLAGS.load_from_npy,
                        load_probe, is_train=False, oh_mapping=mapping)


    # Test model. Using validation since we won't be using real 
    # "test" data yet. Preds will be an array of bounding boxes. 
    start_test = time.time()
    # cls_preds, loc_preds = ssnn.test(X_test)
    cls_preds, loc_preds = ssnn.test(X_test)
    end_test = time.time()

    print("Time to run {} test samples took {} seconds.".format(X_test.shape[0], end_test-start_test))
    
    # Save output.
    save_output(CLS_PREDS, LOC_PREDS, cls_preds, loc_preds, 
                               NUM_HOOK_STEPS, NUM_SCALES, len(ANCHORS), len(CATEGORIES)+1)
    
    cls_f = np.load(CLS_PREDS)
    loc_f = np.load(LOC_PREDS)

    bboxes, bboxes_cls = output_to_bboxes(cls_f, loc_f, NUM_HOOK_STEPS, NUM_SCALES, 
                              DIMS/NUM_HOOK_STEPS, BBOX_PREDS, BBOX_CLS_PREDS, ANCHORS, conf_threshold=0.10)



    # Compute recall and precision.
    compute_mAP(bboxes, bboxes_cls, np.load(BBOX_TEST_LABELS), np.load(CLS_TEST_BBOX), mapping=mapping, threshold=0.25)
    compute_mAP(bboxes, bboxes_cls, np.load(BBOX_TEST_LABELS), np.load(CLS_TEST_BBOX), mapping=mapping, threshold=0.5, plot_category=0)

  
  
# Tensorflow boilerplate code.
if __name__ == '__main__':
  tf.app.run()
