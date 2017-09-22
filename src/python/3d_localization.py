import tensorflow as tf
import numpy as np
from os.path import join, isdir
from os import listdir

flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_string('data_dir', '/home/ryan/cs/datasets/SSNN/test', 'Path to base directory.')
flags.DEFINE_integer('num_epochs', 1, 'Number of epochs to train.')
# ... 

X_NPY_PATH = join(FLAGS.data_dir, 'input_data.npy')
YS_NPY_PATH = join(FLAGS.data_dir, 'segmentation_data.npy')
YL_NPY_PATH = join(FLAGS.data_dir, 'label_data.npy')

def load_data_from_directory(path):
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

  print input_data.shape
  print segmentations.shape
  print labels.shape

  return input_data, segmentations, labels

def load_from_npy(X_path, ys_path, yl_path):
  return np.load(X_path), np.load(ys_path), np.load(yl_path)

    


def main(_):
  X, y_s, y_l = load_data_from_directory(FLAGS.data_dir)
  np.save(X_NPY_PATH, X)
  np.save(YS_NPY_PATH, y_s)
  np.save(YL_NPY_PATH, y_l)


if __name__ == '__main__':
  tf.app.run()
