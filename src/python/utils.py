import numpy as np
import sys
if sys.version_info[0] >= 3:
  from pyntcloud.io import read_ply, write_ply
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

def augment_pointclouds(pointclouds, ys, copies=0):
  for pointcloud, y in list(zip(pointclouds, ys)):
    # Jitter pointclouds
    for _ in range(copies):
      jittered_pc = jitter_pointcloud(pointcloud)
      pointclouds.append(jittered_pc)
      ys.append(y)

  return pointclouds, ys

# Our default is to do 3 equally spaced rotations around the unit circle
def rotate_pointclouds(pointclouds, ys, yl, num_rotations=3):
  delta_r = 2*np.pi / (num_rotations+1)
  rotation_angles = np.linspace(delta_r, 2*np.pi, num_rotations, endpoint=False)

  num_pclouds = len(pointclouds)
  pointclouds = list(pointclouds)
  ys = list(ys)
  yl = list(yl)
  for k in range(num_pclouds):
    shape_pc = pointclouds[k][:, :3]
    color_pc = pointclouds[k][:, 3:]
    for rotation_angle in rotation_angles:
      cosval = np.cos(rotation_angle)
      sinval = np.sin(rotation_angle)
      rotation_matrix = np.array([[cosval, -sinval, 0],
                                  [sinval, cosval, 0],
                                  [0, 0, 1]])
      rotated_pc = np.dot(shape_pc, rotation_matrix)
      pointclouds.append(np.concatenate([rotated_pc, color_pc], axis=-1))
      new_y = []
      for obj in ys[k]:
        rotated_obj = np.dot(obj.reshape((-1, 3)), rotation_matrix)
        LL = np.min(rotated_obj, axis=0)
        UR = np.max(rotated_obj, axis=0)
        new_box = np.concatenate([LL, UR])
        new_y.append(new_box) 

      ys.append(new_y)
      yl.append(yl[k])

  return np.array(pointclouds), np.array(ys), np.array(yl)

def flatten_output(cls_preds, loc_preds, steps, res_factor, num_anchors, num_classes):
  cls_output = []
  loc_output = []
  
  assert len(cls_preds) == len(loc_preds), "Cls and loc prediction arrays are not the same size."

  for scene in range(len(cls_preds)):
    res_factor = 0
    cls_preds_flat = []
    loc_preds_flat = []

    for cls_pred, loc_pred in zip(cls_preds[scene], loc_preds[scene]):
      cls_preds_flat.append(np.reshape(cls_pred, (int((steps/(2**res_factor))**3), num_anchors, num_classes)))
      loc_preds_flat.append(np.reshape(loc_pred, (int((steps/(2**res_factor))**3), num_anchors, 6)))
      res_factor += 1
    cls_output.append(np.concatenate(cls_preds_flat, axis=0))
    loc_output.append(np.concatenate(loc_preds_flat, axis=0))

  cls_output = np.array(cls_output)
  loc_output = np.array(loc_output)
  cls_output = np.apply_along_axis(softmax, 3, cls_output)
 
  return cls_output, loc_output

def softmax(x):
  exp = np.exp(x)
  return exp / np.sum(exp)

def save_output(cls_path, loc_path, cls_preds, loc_preds, steps, res_factor, num_anchors, num_classes):

  cls_output, loc_output = flatten_output(cls_preds, loc_preds, steps, res_factor, num_anchors, num_classes)

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
def nms(cls_preds, loc_preds, overlap_thresh, class_num):

    # coordinates of the bounding boxes
    all_loc_preds = []
    all_cls_preds = []
    num_classes = len(cls_preds[0][0])

    # iterate over rooms  
    for i in range(len(cls_preds)):
        if len(loc_preds[i]) == 0:
            continue
        x1 = loc_preds[i][:,0]
        y1 = loc_preds[i][:,1]
        z1 = loc_preds[i][:,2]
        x2 = loc_preds[i][:,3]
        y2 = loc_preds[i][:,4]
        z2 = loc_preds[i][:,5]
 
        # scores are the probability of a given bbox being an ROI
        scores = cls_preds[i][:,class_num] 
        volumes = np.abs((x2 - x1) * (y2 - y1) * (z2 - z1))
        idxs = np.argsort(scores)[::-1]
        pick = []
        count = 1 

        while len(idxs) > 0:

            # index of the bbox with the highest remaining score
            j = idxs[0]
            pick.append(j)
            xx1 = np.maximum(x1[j], x1[idxs])

            yy1 = np.maximum(y1[j], y1[idxs])
            zz1 = np.maximum(z1[j], z1[idxs])
            xx2 = np.minimum(x2[j], x2[idxs])
            yy2 = np.minimum(y2[j], y2[idxs])
            zz2 = np.minimum(z2[j], z2[idxs])

            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            t = np.maximum(0, zz2 - zz1)
            intersection = w * h * t

            unions = volumes[j] + volumes[idxs] - intersection
    
            # compute the iou
            ious = intersection / unions

            remaining_proposals = np.where(ious <= overlap_thresh)[0]

            if len(remaining_proposals) > 0 and remaining_proposals[0] == 0:
              remaining_proposals = remaining_proposals[1:]

            # delete indices of bboxes that overlap by more than threshold
            idxs = idxs[remaining_proposals] 

        # keep only the bounding boxes that were picked
        all_loc_preds.append(np.array(loc_preds[i][ pick]))
        all_cls_preds.append(np.array(cls_preds[i][ pick]))

    return np.array(all_loc_preds), np.array(all_cls_preds)

def output_to_bboxes(cls_preds, loc_preds, num_steps, num_downsamples, 
                     kernel_size, bbox_path, cls_path, anchor_boxes, conf_threshold=0.5):

  if type(kernel_size) is np.ndarray:
    kernel_size = kernel_size[0]

  all_bboxes = []
  all_cls_vals = []
  for scene in range(cls_preds.shape[0]):
    bboxes = []
    cls_vals = []
    dim = int(num_steps)

    prev_ind = 0
    curr_ksize = kernel_size

    for scale in range(num_downsamples):
      num_categories = cls_preds.shape[3] - 1
      cls_hook = cls_preds[scene, prev_ind:prev_ind+dim**3, :, 1:]
      cls_hook = np.reshape(cls_hook, (dim, dim, dim, len(anchor_boxes), num_categories))
      loc_hook = loc_preds[scene, prev_ind:prev_ind+dim**3]
      loc_hook = np.reshape(loc_hook, (dim, dim, dim, len(anchor_boxes), 6))
      for i in range(dim):
        for j in range(dim):
          for k in range(dim):
            for a, anchor in enumerate(anchor_boxes):
              if max(cls_hook[i, j, k, a]) > conf_threshold:
                center_pt = loc_hook[i, j, k, a, :3] + [i,j,k] + 0.5
                half_dims = (np.exp(loc_hook[i, j, k, a, 3:]) * anchor)/2
                LL = (center_pt - half_dims) * curr_ksize
                UR = (center_pt + half_dims) * curr_ksize
                bbox = np.concatenate([LL, UR], axis=0)
                cls_vals.append(cls_hook[i, j, k, a])
                bboxes.append(bbox)
      prev_ind += dim**3
      dim //= 2
      curr_ksize *= 2  
    all_bboxes.append(np.array(bboxes))
    all_cls_vals.append(np.array(cls_vals))

  all_bboxes = np.array(all_bboxes)
  all_cls_vals = np.array(all_cls_vals)

  if bbox_path is not None:
    print('Saving bbox predictions to {}'.format(bbox_path))

    np.save(bbox_path, all_bboxes)
  if cls_path is not None:
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

def compute_iou(pred, label):
  max_LL = np.max(np.array([pred[:3], label[:3]]), axis=0)
  min_UR = np.min(np.array([pred[3:], label[3:]]), axis=0)
  intersection = max(0, np.prod(min_UR - max_LL))

  union = np.prod(pred[3:]-pred[:3]) + np.prod(label[3:]-label[:3]) - intersection

  if min(min_UR - max_LL) > 0:
    iou = intersection/union
  else:
    iou = 0.0

  return iou

def create_jaccard_labels(labels, categories, num_classes, steps, kernel_size, anchor_boxes, num_downsamples=3, max_dim_thresh=3):
  """
  Args:
    labels (np.array): labeled boxes with shape (batches, box, 6), where the format for
                     a box is min_x, min_y, min_z, max_x, max_y, max_z
    categories (np.array): parallel array to labels with shape (batches, box) with the string name of the box category.
    steps (int): dimension of grid to be explored
    kernel_size (float): size of a grid in meters
    num_downsamples (int): number of hook layers
  """
  cls_labels = []
  loc_labels = []


  for d in range(num_downsamples):
    k = int(steps/(2**d))
    cls_null = np.zeros((len(labels), k, k, k, len(anchor_boxes), num_classes))
    cls_null[:, :, :, :, :, 0] = np.ones((len(labels), k, k, k, len(anchor_boxes)))
    cls_labels.append(cls_null)
    loc_labels.append(np.zeros((len(labels), k, k, k, len(anchor_boxes), 6)))

  for scene_id in range(len(labels)):
    for bbox_id in range(len(labels[scene_id])):
      bbox = labels[scene_id][bbox_id]
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


      if coords[0] >= best_num_steps or coords[1] >= best_num_steps or coords[2] >= best_num_steps or min(coords) < 0:
        print("[WARNING] A bbox label was located outside of the pointcloud.")
        continue

      anchor_ious = []
      best_anchor = None
      best_index = 0
      best_iou = -1

      b1 = bbox_loc - bbox_dims/2
      b2 = bbox_loc + bbox_dims/2
      b = np.append(b1, b2)
      for k, anchor in enumerate(anchor_boxes):
        c1 = coords - anchor/2 + 0.5
        c2 = coords + anchor/2 + 0.5
        c = np.append(c1, c2)
        iou = compute_iou(c,b)
        if iou > best_iou:
          best_anchor = anchor
          best_iou = iou
          best_index = k

      cls_labels[scale][scene_id, coords[0], coords[1], coords[2], best_index] = categories[scene_id][bbox_id]
      loc_labels[scale][scene_id, coords[0], coords[1], coords[2], best_index, :3] = bbox_loc - (coords + 0.5)
      loc_labels[scale][scene_id, coords[0], coords[1], coords[2], best_index, 3:] = np.log(bbox_dims/best_anchor)

      # Second phase: for each feature box, if the jaccard overlap is > 0.25, set it equal to 1 as well.
      
      # Get bbox coords in voxel grid space. This will be divided by 2 every downsample.
      bbox_loc = np.concatenate([bbox[:3] / kernel_size, bbox[3:] / kernel_size], axis=0)
      for s in range(num_downsamples):
        diff = (np.ceil(bbox_loc[3:]) - np.floor(bbox_loc[:3])).astype(int)

        # For each voxel grid the bbox overlaps...
        for i in range(diff[0]):
          for j in range(diff[1]):
            for k in range(diff[2]):
              for a, anchor in enumerate(anchor_boxes):
                # print('start')
                # Get the current coordinate to check.
                curr_coord = np.floor(bbox_loc[:3]).astype(int) + [i,j,k] + 0.5

                # If the current coordinate is outside of the voxel grid, skip it.
                if max(curr_coord -(steps / (2**s))) >= 0:
                  continue

                # Calculate the Jaccard coefficient.
                bbox_LL = bbox_loc[:3]
                bbox_UR = bbox_loc[3:]
                fb_LL = np.array(curr_coord) - anchor/2
                fb_UR = np.array(curr_coord) + anchor/2
                # print('anchor {}'.format(anchor))
                # print('default box: {} {}'.format(fb_LL, fb_UR))
                # print('bbox: {}'.format(bbox_loc))
                if min(fb_UR - bbox_LL) < 0 or min(bbox_UR - fb_LL) < 0:
                  #print('bboxes dont overlap')
                  continue
                max_LL = np.maximum(fb_LL, bbox_LL)
                min_UR = np.minimum(fb_UR, bbox_UR)
                # print('max_LL {} min_UR {}'.format(max_LL, min_UR))
                if min(min_UR - max_LL) < 0.0:
                  intersection = 0
                else:
                  intersection = max(0, np.prod(min_UR - max_LL))
                union = np.prod(fb_UR-fb_LL) + np.prod(bbox_UR-bbox_LL) - intersection
                ji = intersection / union
                

                floored_coord = np.floor(curr_coord).astype(int)
                if ji > 0.25:
                  #cls_labels[s][scene_id, curr_coord[0], curr_coord[1], curr_coord[2]] = 1
                  cls_labels[s][scene_id, floored_coord[0], floored_coord[1], floored_coord[2], a] = categories[scene_id][bbox_id]
                  loc_labels[s][scene_id, floored_coord[0], floored_coord[1], floored_coord[2], a, :3] = (bbox_UR + bbox_LL)/2 - curr_coord
                  loc_labels[s][scene_id, floored_coord[0], floored_coord[1], floored_coord[2], a, 3:] = np.log((bbox_UR - bbox_LL)/anchor)

        bbox_loc /= 2

  # Format into the correct sized array for passing in labels to model.
  cls_labels_flat = []
  loc_labels_flat = []
  res_factor = 0

  for cls_label, loc_label in zip(cls_labels, loc_labels):
    cls_labels_flat.append(np.reshape(cls_label, (-1, int((steps/(2**res_factor))**3), len(anchor_boxes), num_classes)))
    loc_labels_flat.append(np.reshape(loc_label, (-1, int((steps/(2**res_factor))**3), len(anchor_boxes), 6)))
    res_factor += 1

  cls_concat = np.concatenate(cls_labels_flat, axis=1).astype(np.int32)
  loc_concat = np.concatenate(loc_labels_flat, axis=1)
  return cls_concat, loc_concat 

def normalize_pointclouds_stanford(pointcloud_arr, seg_arr, probe_dims):
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

    xyz = (xyz-mins)# / dims * probe_dims

    if gmax is None:
      gmax = maxes
    else:
      gmax = np.array([dims, gmax]).max(axis=0)

    shifted_objs = []
    # Loop through each object label in this scene.
    for obj in seg:
      shifted_objs.append(np.array(obj[:,:3]-mins))# / dims * probe_dims)
    shifted_segmentations.append(shifted_objs)
    new_pc = np.array(xyz)

    new_pc = np.concatenate([new_pc, pointcloud[:, 3:]], axis=1)
    shifted_pointclouds.append(new_pc)
  return shifted_pointclouds, gmax, shifted_segmentations

def normalize_pointclouds_matterport(pointcloud_arr, seg_arr, probe_dims, use_rgb=True):
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
    mult_dims =  probe_dims / dims
    for obj in seg:
      bmins = [mins[0], mins[1], mins[2], mins[0], mins[1], mins[2]]
      shifted_objs.append((obj-bmins) *np.array([mult_dims[0], mult_dims[1], mult_dims[2], mult_dims[0], mult_dims[1], mult_dims[2]]))
    shifted_segmentations.append(shifted_objs)
    
    xyz = (xyz-mins) * mult_dims
    new_pc = np.array(xyz)
    if use_rgb:
      new_pc = np.concatenate([new_pc, pointcloud[:, 3:]], axis=1)
    shifted_pointclouds.append(new_pc)
  return shifted_pointclouds, gmax, shifted_segmentations

def load_points_stanford(path, X_npy_path, ys_npy_path, yl_npy_path,
                load_from_npy=True, areas=None, categories=None):
  """
  Load data from preloaded npy files or from directory.
  """
  if exists(X_npy_path) and load_from_npy:
    assert X_npy_path is not None, "No path given for .npy file."
    print("\tLoading points from npy file...")
    X, ys, yl = load_npy(X_npy_path, ys_npy_path, yl_npy_path)
    new_ds = False
  else:
    assert path is not None, "No path given for pointcloud directory."
    print("\tLoading points from directory...")
    X, ys, yl = load_directory_stanford(path, areas, categories)
    np.save(X_npy_path, X)
    np.save(ys_npy_path, ys)
    np.save(yl_npy_path, yl)
    new_ds = True
  return X, ys, yl, new_ds

def load_points_matterport(path, X_npy_path, yb_npy_path, yl_npy_path,
                           load_from_npy=True, train_test_split=0.9, is_train=True, categories=None, use_rgb=True):
  """
  Load data from preloaded npy files or from directory.
  """
  if exists(X_npy_path) and load_from_npy:
    assert X_npy_path is not None, "No path given for .npy file."
    print("\tLoading points from npy file...")
    X, yb, yl = load_npy(X_npy_path, yb_npy_path, yl_npy_path)
    new_ds = False
  else:
    assert path is not None, "No path given for pointcloud directory."
    print("\tLoading points from directory...")
    X, yb, yl = load_directory_matterport(path, train_test_split, is_train, categories, use_rgb)
    np.save(X_npy_path, X)
    np.save(yb_npy_path, yb)
    np.save(yl_npy_path, yl)
    new_ds = True
  return X, yb, yl, new_ds

def load_directory_matterport(path, train_test_split, is_train, objects, use_rgb=True):
  """
  Loads pointclouds from matterport dataset.

  Assumes dataset structure is as follows:
  base
    building_name
      processed_regions
        region0.npy
        region0_bboxes.npy
        region0_labels.npy
        ...
    ...
  """
  all_areas = sorted(listdir(path))

  if is_train:
    areas = all_areas[:int(len(all_areas)*train_test_split)]
    #areas = all_areas[int(len(all_areas)*(.65)):]
  else:
    areas = all_areas[int(len(all_areas)*train_test_split):]
    #areas = all_areas[:int(len(all_areas)*.05)]

  input_data = []
  bboxes = []
  labels = []
  total_regions = 0
  # Loop through buildings
  if areas is None:
    areas = sorted(listdir(path))
  for area in areas:
    print("\t\tLoading area {}...".format(area))
    area_path = join(path, area, "processed_regions")
    if not isdir(area_path):
      continue
      
    ri = 0
    while exists(join(area_path, "region{}.ply".format(ri))):
      room = "region{}".format(ri)
      ri += 1
      room_path = join(area_path, room)

      # print("\tLoading room {}...".format(room))

      # Load point cloud
      categories = np.load(room_path+"_labels.npy")

      input_pc = read_ply(room_path+".ply")
      bbox = np.load(room_path+"_bboxes.npy")
      input_pc = input_pc["points"].as_matrix(columns=["x", "y", "z", "r", "g", "b"])

      fbbox = []
      flabel = []
      matches = 0
      for ibbox, ilabel in zip(bbox, categories):
        if len(objects) == 0 or ilabel in objects:
          fbbox.append(ibbox)
          flabel.append(ilabel)
          matches += 1
      
      if matches > 0:
        bboxes.append(fbbox)
        labels.append(flabel)
        input_data.append(input_pc)

    print("\t\tLoaded {} regions from area {}".format(ri, area))
    total_regions += ri

  input_data = np.array(input_data)
  bboxes = np.array(bboxes)
  labels = np.array(labels)

  return input_data, bboxes, labels


def load_points_sunrgbd(path, X_npy_path, yb_npy_path, yl_npy_path,
                           load_from_npy=True, train_test_split=0.9, is_train=True, categories=None, use_rgb=True):
  """
  Load data from preloaded npy files or from directory.
  """
  if exists(X_npy_path) and load_from_npy:
    assert X_npy_path is not None, "No path given for .npy file."
    print("\tLoading points from npy file...")
    X, yb, yl = load_npy(X_npy_path, yb_npy_path, yl_npy_path)
    new_ds = False
  else:
    assert path is not None, "No path given for pointcloud directory."
    print("\tLoading points from directory...")
    X, yb, yl, images, depthmaps, Ks, RTs = load_directory_sunrgbd(path, train_test_split, is_train, categories, use_rgb)

    np.save(X_npy_path, X)
    np.save(yb_npy_path, yb)
    np.save(yl_npy_path, yl)
    new_ds = True
  return X, yb, yl, new_ds, images, depthmaps, Ks, RTs

def load_directory_sunrgbd(path, train_test_split, is_train, objects, use_rgb=True):
  """
  Loads pointclouds from matterport dataset.

  Assumes dataset structure is as follows:
  base
    building_name
      processed_regions
        region0.npy
        region0_bboxes.npy
        region0_labels.npy
        ...
    ...
  """

  all_areas = sorted(listdir(path))
  all_areas = ['kv1', 'kv2']

  if is_train:
    #areas = all_areas[:int(len(all_areas)*train_test_split)]
    areas = all_areas
  else:
    areas = all_areas[int(len(all_areas)*train_test_split):]
    #areas = all_areas[:int(len(all_areas)*.05)]

  input_data = []
  bboxes = []
  labels = []
  images = []
  depthmaps = []
  Ks = []
  RTs = []
  total_regions = 0

  # Loop through buildings
  if areas is None:
    areas = sorted(listdir(path))

  for area in areas:
    ri = 0
    counter = 0
    print("\t\tLoading area {}...".format(area))
    for dataset in listdir(join(path, area)):
      area_path = join(path, area, dataset+'_processed')

      if not isdir(area_path):
        continue
      
      while exists(join(area_path, "region{}.ply".format(ri))):
        room = "region{}".format(ri)
        ri += 1
        room_path = join(area_path, room)

        # print("\tLoading room {}...".format(room))

        # Load point cloud
        categories = np.load(room_path+"_labels.npy")
        rgb_img = np.load(room_path+"_rgb.npy")
        depthmap = np.load(room_path+"_d.npy")
        input_pc = read_ply(room_path+".ply")
        bbox = np.load(room_path+"_bboxes.npy")
        K = np.load(room_path+"_k.npy")
        RT = np.load(room_path+"_rt.npy")
        input_pc = input_pc["points"].as_matrix(columns=["x", "y", "z", "r", "g", "b"])

        fbbox = []
        flabel = []
        matches = 0
        for ibbox, ilabel in zip(bbox, categories):
          if len(objects) == 0 or ilabel in objects:
            fbbox.append(ibbox)
            flabel.append(ilabel)
            matches += 1
        
        if matches > 0:
          bboxes.append(fbbox)
          labels.append(flabel)
          input_data.append(input_pc)
          images.append(rgb_img)
          depthmaps.append(depthmap)
          Ks.append(K)
          RTs.append(RT)
          counter += 1

    print("\t\tLoaded {} regions from area {}".format(counter, area))
    total_regions += counter

  input_data = np.array(input_data)
  bboxes = np.array(bboxes)
  labels = np.array(labels)
  images = np.array(images)
  depthmaps = np.array(depthmaps)
  Ks = np.array(Ks)
  RTs = np.array(RTs)

  return input_data, bboxes, labels, images, depthmaps, Ks, RTs

  

def load_directory_stanford(path, areas, categories):
  """
  Loads pointclouds from Stanford dataset.

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

      # Loop and load Annotations folder
      annotation_pc = []
      annotation_label = []
      for annotation in listdir(join(room_path, 'Annotations')):
        if (not categories is None) and annotation.split('_')[0] not in categories:
          continue
        annotation_pc.append(np.genfromtxt(
                  join(room_path, 'Annotations', annotation), dtype=np.float32))
        annotation_label.append(annotation.split('.')[0])
      if len(annotation_pc) != 0:
        # Load point cloud
        input_pc = np.genfromtxt(join(room_path, room+'.txt'), dtype=np.float32)
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


def one_hot_vectorize_categories(yl, mapping=None):
  """
  Converts category array into one-hot vectored labels.
  """
  if mapping is None:
    # Loop through all labels to get a set of the labels
    mapping = dict()
    curr_index = 0
    for i in range(len(yl)):
      pc = yl[i]
      for obj in pc:
        if '_' in obj:
          obj = obj[:obj.index('_')]
        if obj not in mapping:
          mapping[obj] = curr_index
          curr_index += 1
  else:
    classes = set()
    for i in range(len(yl)):
      pc = yl[i]
      for obj in pc:
        if '_' in obj:
          obj = obj[:obj.index('_')]
        classes.add(obj)
    num_missing_classes = len(mapping) - len(classes)
    if num_missing_classes > 0:
      print("\t[WARNING] Test set doesn't have some of the selected classes that the train set has.")
    elif num_missing_classes < 0:
      print("\t[WARNING] Training set doesn't have some of the selected classes that the test set has.")

  # num_classes is the number of categories plus the null class.  
  num_classes = len(mapping)+1

  # Loop through the labels again and convert to one-hot vector
  onehot_labels = []
  for i in range(len(yl)):
    pc = yl[i]
    pc_objs = []
    for obj in pc:
      if '_' in obj:
        obj = obj[:obj.index('_')]
      onehot = np.zeros(num_classes)
      # Reserve the 0th index for the null class.
      onehot[mapping[obj]+1] = 1
      pc_objs.append(onehot)
    onehot_labels.append(pc_objs)

  print('\tDictionary for class mapping: {}'.format(mapping))

  return np.array(onehot_labels), mapping
  

if __name__ == '__main__':
  cls_preds = np.load('/home/ryan/cs/datasets/SSNN/matterport/v1/scans/outputs/cls_predictions.npy')
  loc_preds = np.load('/home/ryan/cs/datasets/SSNN/matterport/v1/scans/outputs/loc_predictions.npy')

  output_to_bboxes(cls_preds, loc_preds, 16, 3, 
                     .46875, '/home/ryan/cs/datasets/SSNN/matterport/v1/scans/outputs/bbox_predictions.npy', '/home/ryan/cs/datasets/SSNN/buildings/outputs/bbox_cls_predictions.npy')
