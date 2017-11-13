import numpy as np
import sys
import paths
from os.path import join, exists, isdir, isfile
from os import makedirs, listdir
import time
from scipy.misc import imsave

def softmax(x):
  exp = np.exp(x)
  return exp / np.sum(exp)

def save_output(cls_path, loc_path, cls_preds, loc_preds, steps, res_factor):
  cls_output = []
  loc_output = []
  assert len(cls_preds) == len(loc_preds), "Cls and loc prediction arrays are not the same size."
  for scene in range(len(cls_preds)):
    res_factor = 0
    cls_preds_flat = []
    loc_preds_flat = []
    for cls_pred, loc_pred in zip(cls_preds[scene], loc_preds[scene]):
      cls_preds_flat.append(np.reshape(cls_pred, ((steps/(2**res_factor))**3, 2)))
      loc_preds_flat.append(np.reshape(loc_pred, ((steps/(2**res_factor))**3, 6)))
      res_factor += 1
    cls_output.append(np.concatenate(cls_preds_flat, axis=0))
    loc_output.append(np.concatenate(loc_preds_flat, axis=0))

  cls_output = np.array(cls_output)
  loc_output = np.array(loc_output)

  print cls_output.shape
  cls_output = np.apply_along_axis(softmax, 2, cls_output)
  print cls_output.shape
  print('Saving cls predictions to {}'.format(cls_path))
  np.save(cls_path, cls_output)
  print('Saving loc predictions to {}'.format(loc_path))
  np.save(loc_path, loc_output)

def voxelize_labels(labels, steps, kernel_size):
  """
  Args:
    labels (tensor): labeled boxes with (batches, box, 6), with the format for
                     a box being min_x, min_y, min_z, max_x, max_y, max_z
    steps (int): dimension of grid to be explored
    kernel_size (float): size of a grid in meters
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

            intersection = np.prod(UR-LL)

            if coords[0] >= steps or coords[1] >= steps or coords[2] >= steps:
              continue

            prev_val = vox_label[scene_id, coords[0], coords[1], coords[2]]
            
            vox_label[scene_id, coords[0], coords[1], coords[2]] = \
                    np.max([intersection, prev_val])
  return vox_label

def create_jaccard_labels(labels, steps, kernel_size, num_downsamples=3):
  """
  Args:
    labels (tensor): labeled boxes with (batches, box, 6), with the format for
                     a box being min_x, min_y, min_z, max_x, max_y, max_z
    steps (int): dimension of grid to be explored
    kernel_size (float): size of a grid in meters
    num_downsamples (int): number of hook layers
  """
  cls_labels = []
  loc_labels = []
  for d in range(num_downsamples):
    cls_labels.append(np.zeros((len(labels), steps/(2**d), steps/(2**d), steps/(2**d))))
    loc_labels.append(np.zeros((len(labels), steps/(2**d), steps/(2**d), steps/(2**d), 6)))

  
  for scene_id in range(len(labels)):
    # First phase: for each GT box, set the closest feature box to 1.
    for bbox in labels[scene_id]:
      # bbox is [min_x, min_y, min_z, max_x, max_y, max_z]
      bbox_dims = (bbox[3:] - bbox[:3]) / kernel_size
      bbox_loc = (bbox[3:] + bbox[:3]) / 2 / kernel_size
      max_dim = np.max(bbox_dims)
      scale = 0
      for _ in range(num_downsamples-1):
        if max_dim < 10:
          break
        max_dim /= 2
        bbox_dims /= 2
        bbox_loc /= 2
        scale += 1

      best_kernel_size = kernel_size * 2**scale
      best_num_steps = steps / (2**scale)

      coords = np.trunc(bbox_loc).astype(int)

      if coords[0] >= best_num_steps or coords[1] >= best_num_steps or coords[2] >= best_num_steps:
        continue

      cls_labels[scale][scene_id, coords[0], coords[1], coords[2]] = 1
      loc_labels[scale][scene_id, coords[0], coords[1], coords[2], :3] = bbox_loc - coords
      loc_labels[scale][scene_id, coords[0], coords[1], coords[2], 3:] = bbox_dims

    # Second phase: for each feature box, if the jaccard overlap is > 0.5, set it equal to 1 as well.
    """
    for s in range(num_downsamples):
      for i in range(steps):
        for j in range(steps):
          for k in range(steps):
            for bbox in labels[scene_id]:
              bbox_loc = bbox / 2 / (2**s)
              bbox_LL = bbox_loc[:3]/ kernel_size
              bbox_UR = bbox_loc[3:]/ kernel_size
              fb_LL = np.array([i, j, k])
              fb_UR = np.array([i+1, j+1, k+1])
              if max(fb_UR - bbox_LL) < 0 or min(bbox_UR - fb_LL) < 0:
                continue
              # print fb_UR
              # print bbox_LL
              # print bbox_UR
              # print fb_LL
              max_UR = np.maximum(fb_UR, bbox_UR)
              max_LL = np.maximum(fb_LL, bbox_LL)
              min_UR = np.minimum(fb_UR, bbox_UR)
              min_LL = np.minimum(fb_LL, bbox_LL)
              ji = np.prod(min_UR - max_LL) / np.prod(max_UR - min_LL)
              if ji > 0.5:
                cls_labels[scale][scene_id, i, j, k] = 1
                loc_labels[scale][scene_id, i, j, k, :3] = (bbox_UR + bbox_LL)/2 - [i,j,k]
                loc_labels[scale][scene_id, i, j, k, 3:] = bbox_UR - bbox_LL - [i,j,k]
  """
  cls_labels_flat = []
  loc_labels_flat = []
  res_factor = 0

  for cls_label, loc_label in zip(cls_labels, loc_labels):
    cls_labels_flat.append(np.reshape(cls_label, (-1, (steps/(2**res_factor))**3, 1)))
    loc_labels_flat.append(np.reshape(loc_label, (-1, (steps/(2**res_factor))**3, 6)))
    res_factor += 1

  cls_concat = np.concatenate(cls_labels_flat, axis=1).astype(np.int32)
  cls_no_class = np.ones_like(cls_concat) - cls_concat
  cls_concat = np.concatenate([cls_no_class, cls_concat], axis=-1)

  loc_concat = np.concatenate(loc_labels_flat, axis=1)
  return cls_concat, loc_concat 

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
  else:
    assert path is not None, "No path given for pointcloud directory."
    print("Loading points from directory...")
    X, ys, yl = load_directory(path)
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
  for area in sorted(listdir(path)):
    print("Loading area {}...".format(area))
    area_path = join(path, area)
    if not isdir(area_path):
      continue
      
    # Loop through rooms
    for room in sorted(listdir(area_path)):
      room_path = join(area_path, room)
      if not isdir(room_path) or room.endswith('Angle.txt') or \
         room == '.DS_Store':
        continue
      print "\tLoading room {}...".format(room)

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
