import numpy as np
import sys
import paths
from os.path import join, exists, isdir, isfile
from os import makedirs
import time
from scipy.misc import imsave



def save_output(output_path, output):
  print('Saving predictions to {}'.format(output_path))
  np.save(output_path, output)


def voxelize_labels(labels, steps, kernel_size):
  """
  Args:
    preds (tensor): predicted confidence value for a certain box (batches, x, y, z)
    labels (tensor): labeled boxes with (batches, box, 6), with the format for
                     a box being min_x, min_y, min_z, max_x, max_y, max_z
  """
  vox_label = np.zeros((len(labels), steps, steps, steps))

  for scene_id in range(len(labels)):
    for bbox in labels[scene_id]:
      # bbox is [min_x, min_y, min_z, max_x, max_y, max_z]
      c1 = np.floor(bbox[:3] / kernel_size).astype(int)
      c2 = np.ceil(bbox[3:] / kernel_size).astype(int)
      diff = c2 - c1
      for i in range(diff[0]):
        for j in range(diff[1]):
          for k in range(diff[2]):
            coords = c1 + [i,j,k]
            
            LL = np.max([bbox[:3]/kernel_size, coords], axis=0)
            UR = np.min([bbox[3:]/kernel_size, coords+1], axis=0) 
            intersection = np.sqrt(np.sum(np.square([LL, UR])))

            prev_val = vox_label[scene_id, coords[0], coords[1], coords[2]]
            vox_label[scene_id, coords[0], coords[1], coords[2]] = \
                    np.max([intersection, prev_val])

  return vox_label

def normalize_pointclouds(pointcloud_arr, seg_arr):
  """
  Shifts pointclouds so the smallest xyz values map to the origin. We want to 
  preserve relative scale, but this also leads to excess or missed computation
  between rooms. 

  TODO: allow layers to input a dynamic dimension (requires changes number of 
  steps as well)

  Args:
    pointcloud_arr (np.ndarray): array of all pointclouds with shape 
                                 (clouds, points, 6)
  """
  # Find the minimum x, y, and z values.
  shifted_pointclouds = []
  shifted_segmentations = []
  gmax = None

  # Loop through scenes.
  for pointcloud, seg in zip(pointcloud_arr, seg_arr):
    xyz = pointcloud[:, :3]
    mins = xyz.min(axis=0)
    maxes = xyz.max(axis=0)
    dims = maxes-mins
    if gmax is None:
      gmax = maxes
    else:
      gmax = np.array([dims, gmax]).max(axis=0)

    shifted_objs = []
    # Loop through each object label in this scene.
    for obj in seg:
      shifted_objs.append(np.array(obj[:,:3]-mins))
    shifted_segmentations.append(shifted_objs)

    shifted_pointclouds.append(np.array([xyz-mins]))
  return shifted_pointclouds, gmax, shifted_segmentations

def load_points(path, X_npy_path, ys_npy_path, yl_npy_path,
                load_from_npy=True):
  """
  Load data from preloaded npy files or from directory.
  """
  if exists(X_npy_path) and load_from_npy:
    assert X_npy_path is not None, "No path given for .npy file."
    print("Loading points from npy file...")
    X, ys, yl = load_npy(X_npy_path, ys_npy_path, yl_npy_path)
    # np.save(X_npy_path, X[:10])
    # np.save(ys_npy_path, ys[:10])
    # np.save(yl_npy_path, yl[:10])
  else:
    assert path is not None, "No path given for pointcloud directory."
    print("Loading points from directory...")
    X, ys, yl = load_directory(FLAGS.data_dir)
    np.save(X_npy_path, X)
    np.save(ys_npy_path, ys)
    np.save(yl_npy_path, yl)
  return X, ys, yl

def load_directory(path):
  """
  Loads pointclouds from dataset.

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
      if not isdir(room_path) or room.endswith('Angle.txt') or \
         room == '.DS_Store':
        continue
        
      # Load point cloud
      input_pc = np.loadtxt(join(room_path, room+'.txt'), dtype=np.float32)
      
      # Loop and load Annotations folder
      annotation_pc = []
      annotation_label = []
      for annotation in listdir(join(room_path, 'Annotations')):
        if annotation.startswith('wall') or annotation.startswith('ceiling') \
                                         or not annotation.endswith('.txt'):
          continue
        annotation_pc.append(np.loadtxt(
                  join(room_path, 'Annotations', annotation), dtype=np.float32))
        annotation_label.append(annotation.split('.')[0])
      annotation_pc = np.array(annotation_pc)
      
      input_data.append(input_pc)
      segmentations.append(annotation_pc)
      labels.append(annotation_label)

  input_data = np.array(input_data)
  segmentations = np.array(segmentations)
  labels = np.array(labels)

  return input_data, segmentations, labels

def load_npy(X_path, ys_path, yl_path):
  """
  Loads dataset from pre-loaded path if available.
  """
  assert exists(X_path), "Train npy file (X) does not exist."
  assert exists(ys_path), "Train npy file (ys) does not exist."
  assert exists(yl_path), "Train npy file (yl) does not exist."
  return np.load(X_path), np.load(ys_path), np.load(yl_path)