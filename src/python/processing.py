import numpy as np
import os
from os.path import join, isdir, exists
from os import listdir, makedirs
from utils import *
from load_data import *
import time
from object_boundaries import generate_bounding_boxes
import os
import psutil
import pickle as pkl
from matplotlib.colors import rgb_to_hsv


# returns None if it is batch loading
def process_rgb2hsv(X):

  print("\tConverting RGB to HSV...")

  for i in range(len(X)):
    X[i][:, 3:] = rgb_to_hsv(X[i][:,3:])
  return X



def process_bounding_boxes(yb_raw, bbox_labels, ds_name):
  if ds_name == 'stanford':
    print("\tGenerating bboxes...")
    bboxes = generate_bounding_boxes(yb_raw, bbox_labels)
  else:
    bboxes = yb_raw

  return bboxes

# Our default is to do 3 equally spaced rotations around the unit circle
def rotate_pointclouds(pointclouds, ys, yl, num_rotations=3):
  print("\tAugmenting dataset...")
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
      rotated_pc = np.dot(shape_pc, rotation_matrix.T)
      pointclouds.append(np.concatenate([rotated_pc, color_pc], axis=-1))
      new_y = []
      num_90 = int(rotation_angle / (np.pi / 2))
      if num_90 % 2 == 1:
        turn_90 = True
      else:
        turn_90 = False
      extra_theta = rotation_angle % (np.pi / 2)

      for obj in ys[k]:
        #rotated_obj = np.dot(obj.reshape((-1, 3)), rotation_matrix)
        rotated_centroid = np.dot(obj[:3], rotation_matrix.T)
        if turn_90:
          rotated_coeffs = np.array([obj[4], obj[3], obj[5]])
        else:
          rotated_coeffs = obj[3:6]

        if len(obj) > 6:
          rotated_theta = np.array([obj[6]+extra_theta])

          # LL = np.min(rotated_obj, axis=0)
          # UR = np.max(rotated_obj, axis=0)
          new_box = np.concatenate([rotated_centroid, rotated_coeffs, rotated_theta])
        else:
          new_box = np.concatenate([rotated_centroid, rotated_coeffs])
        new_y.append(new_box) 

      ys.append(new_y)
      yl.append(yl[k])
  return np.array(pointclouds), np.array(ys), np.array(yl)



def normalize_pointclouds(pointcloud_arr, seg_arr, probe_dims, transforms, use_rgb=True, dataset='sunrgbd'):
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
  print("\tNormalizing pointclouds...")
  shifted_pointclouds = []
  shifted_segmentations = []
  gmax = None

  
  # Loop through scenes.
  for i in range(len(pointcloud_arr)):#zip(pointcloud_arr, seg_arr):
    pointcloud = pointcloud_arr[i]
    if len(seg_arr) > 0:
      seg = seg_arr[i]
    else:
      seg = None
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
    dims[dims < 0.1] = 0.1
    mult_dims =  probe_dims / dims
    if seg is not None:
      for obj in seg:
        #new_bbox = np.concatenate([obj[:3]-mins, obj[3:6]*mult_dims, obj[6:]], axis=0)
        if dataset == 'sunrgbd':
          new_bbox = np.concatenate([obj[:3]-mins, obj[3:6], obj[6:]], axis=0)
        else:
          # print('start')
          # print(obj)
          center = (obj[3:6]+obj[:3])/2
          side_dims = (obj[3:6]-obj[:3])/2
          new_bbox = np.concatenate([center-mins, side_dims, obj[6:]], axis=0)
          # print(new_bbox)
        shifted_objs.append(new_bbox)
      shifted_segmentations.append(shifted_objs)
    
    #xyz = (xyz-mins) * mult_dims
    xyz = xyz-mins
    transforms['t'].append(mins)
    transforms['s'].append(mult_dims)


    new_pc = np.array(xyz)
    if use_rgb:
      new_pc = np.concatenate([new_pc, pointcloud[:, 3:]], axis=1)
    shifted_pointclouds.append(new_pc)

  
  return shifted_pointclouds, gmax, shifted_segmentations, transforms

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
  print("\tNormalizing pointclouds...")
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
