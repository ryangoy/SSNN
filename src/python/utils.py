import numpy as np
import sys
import paths
from os.path import join, exists, isdir, isfile, basename
from os import makedirs, listdir
import time
from scipy.misc import imsave
import matplotlib.pyplot as plt

# From PointNet
def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.05):
  B, N, C = pointcloud.shape
  jittered_pc = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
  jittered_pc += pointcloud
  return jittered_pc

def augment_pointclouds(pointclouds, ys, copies=2):
  for pointcloud, y in zip(pointclouds, ys):
    # Jitter pointclouds
    for _ in range(copies):
      jittered_pc = jitter_pointcloud(pointcloud)
      pointclouds.append(jittered_pc)
      ys.append(y)


  return pointclouds, ys

def flatten_output(cls_preds, loc_preds, steps, res_factor):
  cls_output = []
  loc_output = []
  
  assert len(cls_preds) == len(loc_preds), "Cls and loc prediction arrays are not the same size."

  for scene in range(len(cls_preds)):
    res_factor = 0
    cls_preds_flat = []
    loc_preds_flat = []
    for cls_pred, loc_pred in zip(cls_preds[scene], loc_preds[scene]):
      cls_preds_flat.append(np.reshape(cls_pred, (int((steps/(2**res_factor))**3), 2)))
      loc_preds_flat.append(np.reshape(loc_pred, (int((steps/(2**res_factor))**3), 6)))
      res_factor += 1
    cls_output.append(np.concatenate(cls_preds_flat, axis=0))
    loc_output.append(np.concatenate(loc_preds_flat, axis=0))

  cls_output = np.array(cls_output)
  loc_output = np.array(loc_output)
  cls_output = np.apply_along_axis(softmax, 2, cls_output)
 
  return cls_output, loc_output

def softmax(x):
  exp = np.exp(x)
  return exp / np.sum(exp)

def save_output(cls_path, loc_path, cls_preds, loc_preds, steps, res_factor):

  cls_output, loc_output = flatten_output(cls_preds, loc_preds, steps, res_factor)

  print('Saving cls predictions to {}'.format(cls_path))
  np.save(cls_path, cls_output)
  print('Saving loc predictions to {}'.format(loc_path))
  np.save(loc_path, loc_output)
  # nms_output = nms(cls_output, loc_output, 0.75, steps, res_factor)
  # print('Original number of loc predictions', [len(i) for i in loc_output])
  # print('nms number of loc predictions', [len(i) for i in nms_output])
  # print('Saving loc predictions (post nms) to {}'.format(nms_path))
  #np.save(nms_path, nms_output)
  return cls_output, loc_output

# adapted from https://github.com/rbgirshick/voc-dpm/blob/master/test/nms.m
def nms(cls_preds, loc_preds, overlap_thresh, steps, res_factor, needs_flattening=False):
    # coordinates of the bounding boxes
    if needs_flattening:
        cls_preds, loc_preds = flatten_output(cls_preds, loc_preds, steps, res_factor)

    all_loc_preds = []

    # iterate over rooms    
    for i in range(len(cls_preds)):
        x1 = loc_preds[i,:,0]
        y1 = loc_preds[i,:,1]
        z1 = loc_preds[i,:,2]
        x2 = loc_preds[i,:,3]
        y2 = loc_preds[i,:,4]
        z2 = loc_preds[i,:,5]
 
        # scores are the probability of a given bbox being an ROI
        score = cls_preds[i,:,1] 
        volume = (x2 - x1 + 1) * (y2 - y1 + 1) * (z2 - z1 + 1)
        idxs = np.argsort(score)
        pick = []
        count = 1 

        while len(idxs) > 0:

            # index of the bbox with the highest remaining score
            last = len(idxs) - 1
            j = idxs[last]
            pick.append(j)
 
            xx1 = np.maximum(x1[j], x1[idxs[:last]])
            yy1 = np.maximum(y1[j], y1[idxs[:last]])
            zz1 = np.maximum(z1[j], z1[idxs[:last]])
            xx2 = np.minimum(x2[j], x2[idxs[:last]])
            yy2 = np.minimum(y2[j], y2[idxs[:last]])
            zz2 = np.minimum(z2[j], z2[idxs[:last]])

            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            t = np.maximum(0, zz2 - zz1 + 1)
 
            # compute the ratio of overlap
            o = (w * h * t) / volume[idxs[:last]]
 
            # delete indices of bboxes that overlap by more than threshold
            idxs = np.delete(idxs, np.concatenate(([last],
                np.where(o > overlap_thresh)[0])))
 
        # keep only the bounding boxes that were picked
        all_loc_preds.append(np.array(loc_preds[i, pick]))

    return np.array(all_loc_preds)

def output_to_bboxes(cls_preds, loc_preds, num_steps, num_downsamples, 
                     kernel_size, bbox_path, cls_path, conf_threshold=0.4):
  all_bboxes = []
  all_cls_vals = []
  for scene in range(cls_preds.shape[0]):
    bboxes = []
    cls_vals = []
    dim = num_steps

    prev_ind = 0
    curr_ksize = kernel_size
    for scale in range(3):
      cls_hook = cls_preds[scene, prev_ind:prev_ind+dim**3, 1]
      cls_hook = np.reshape(cls_hook, (dim, dim, dim))
      loc_hook = loc_preds[scene, prev_ind:prev_ind+dim**3]
      loc_hook = np.reshape(loc_hook, (dim, dim, dim, 6))
      for i in range(dim):
        for j in range(dim):
          for k in range(dim):
            if cls_hook[i, j, k] > conf_threshold:
              center_pt = np.array([i, j, k]) + loc_hook[i, j, k, :3]
              half_dims = (loc_hook[i, j, k, 3:]+1)/2
              LL = (center_pt - half_dims) * curr_ksize
              UR = (center_pt + half_dims) * curr_ksize
              bbox = np.concatenate([LL, UR], axis=0)
              cls_vals.append(cls_hook[i, j, k])
              bboxes.append(bbox)
      prev_ind += dim**3
      dim //= 2
      curr_ksize *= 2  
    all_bboxes.append(np.array(bboxes))
    all_cls_vals.append(np.array(cls_vals))

  all_bboxes = np.array(all_bboxes)
  all_cls_vals = np.array(all_cls_vals)

  print('Saving bbox predictions to {}'.format(bbox_path))
  np.save(bbox_path, all_bboxes)
  print('Saving bbox cls predictions to {}'.format(cls_path))
  np.save(cls_path, all_cls_vals)

  return all_bboxes, all_cls_vals

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



def create_jaccard_labels(labels, steps, kernel_size, num_downsamples=3, max_dim_thresh=3):
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
    k = int(steps/(2**d))
    cls_labels.append(np.zeros((len(labels), k, k, k)))
    loc_labels.append(np.zeros((len(labels), k, k, k, 6)))

  
  for scene_id in range(len(labels)):
    for bbox in labels[scene_id]:

      # First phase: for each GT box, set the closest feature box to 1.

      # bbox is [min_x, min_y, min_z, max_x, max_y, max_z]
      bbox_dims = (bbox[3:] - bbox[:3]) / kernel_size
      bbox_loc = ((bbox[3:] + bbox[:3]) / 2) / kernel_size
      max_dim = np.max(bbox_dims)
      scale = 0

      for _ in range(num_downsamples-1):
        if max_dim < max_dim_thresh:
          break
        max_dim /= 2
        bbox_dims /= 2
        bbox_loc /= 2
        scale += 1
      best_kernel_size = kernel_size * 2**scale
      best_num_steps = steps / (2**scale)
      coords = np.floor(bbox_loc).astype(int)

      if coords[0] >= best_num_steps or coords[1] >= best_num_steps or coords[2] >= best_num_steps:
        continue

      cls_labels[scale][scene_id, coords[0], coords[1], coords[2]] = 1
      loc_labels[scale][scene_id, coords[0], coords[1], coords[2], :3] = bbox_loc - coords
      loc_labels[scale][scene_id, coords[0], coords[1], coords[2], 3:] = bbox_dims - 1

      # Second phase: for each feature box, if the jaccard overlap is > 0.5, set it equal to 1 as well.
      # This is kind of hacky for now until I debug it.
      bbox_loc = np.concatenate([bbox[:3] / kernel_size, bbox[3:] / kernel_size], axis=0)
      for s in range(num_downsamples):
        diff = (np.ceil(bbox_loc[3:]) - np.floor(bbox_loc[:3])).astype(int)
        for i in range(diff[0]):
          for j in range(diff[1]):
            for k in range(diff[2]):
              curr_coord = np.floor(bbox_loc[:3]).astype(int) + [i,j,k]
              if max(curr_coord -(steps / (2**s))) >= 0:
                continue
              bbox_LL = bbox_loc[:3]
              bbox_UR = bbox_loc[3:]
              fb_LL = np.array(curr_coord)
              fb_UR = np.array(curr_coord+1)
              if min(fb_UR - bbox_LL) < 0 or min(bbox_UR - fb_LL) < 0:
                continue
              max_UR = np.maximum(fb_UR, bbox_UR)
              max_LL = np.maximum(fb_LL, bbox_LL)
              min_UR = np.minimum(fb_UR, bbox_UR)
              min_LL = np.minimum(fb_LL, bbox_LL)
              ji = np.prod(min_UR - max_LL) / np.prod(max_UR - min_LL)

              if ji > 0.1:
                cls_labels[s][scene_id, curr_coord[0], curr_coord[1], curr_coord[2]] = 1
                loc_labels[s][scene_id, curr_coord[0], curr_coord[1], curr_coord[2], :3] = (bbox_UR + bbox_LL)/2 - [i,j,k]
                loc_labels[s][scene_id, curr_coord[0], curr_coord[1], curr_coord[2], 3:] = bbox_UR - bbox_LL - 1
        bbox_loc /= 2

  # Format into the correct sized array for passing in labels to model.
  cls_labels_flat = []
  loc_labels_flat = []
  res_factor = 0

  for cls_label, loc_label in zip(cls_labels, loc_labels):
    cls_labels_flat.append(np.reshape(cls_label, (-1, int((steps/(2**res_factor))**3), 1)))
    loc_labels_flat.append(np.reshape(loc_label, (-1, int((steps/(2**res_factor))**3), 6)))
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
                load_from_npy=True, areas=None):
  """
  Load data from preloaded npy files or from directory.
  """
  if exists(X_npy_path) and load_from_npy:
    assert X_npy_path is not None, "No path given for .npy file."
    print("Loading points from npy file...")
    X, ys, yl = load_npy(X_npy_path, ys_npy_path, yl_npy_path)
    new_ds = False
  else:
    assert path is not None, "No path given for pointcloud directory."
    print("Loading points from directory...")
    X, ys, yl = load_directory(path, areas)
    np.save(X_npy_path, X)
    np.save(ys_npy_path, ys)
    np.save(yl_npy_path, yl)
    new_ds = True
  return X, ys, yl, new_ds

def load_directory(path, areas):
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
  if areas is None:
    areas = sorted(listdir(path))
  for area in areas:
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
      print("\tLoading room {}...".format(room))

      # Load point cloud
      input_pc = np.genfromtxt(join(room_path, room+'.txt'), dtype=np.float32)
      
      # Loop and load Annotations folder
      annotation_pc = []
      annotation_label = []
      for annotation in listdir(join(room_path, 'Annotations')):
        if annotation.startswith('wall') or annotation.startswith('ceiling') or\
           annotation.startswith('beam') or annotation.startswith('floor') or\
           annotation.startswith('door') or not annotation.endswith('.txt'):
          continue
        annotation_pc.append(np.genfromtxt(
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

if __name__ == '__main__':
  cls_preds = np.load('cls_predictions.npy')
  loc_preds = np.load('loc_predictions.npy')

  output_to_bboxes(cls_preds, loc_preds, 16, 3, 
                     .46875, 'bbox_predictions.npy', 'bbox_cls_predictions.npy')