import numpy as np
import sys
if sys.version_info[0] >= 3:
  from pyntcloud.io import read_ply, write_ply
from os.path import join, exists, isdir, isfile, basename
from os import makedirs, listdir
import time
from scipy.misc import imsave
import matplotlib.pyplot as plt
import pandas as pd
import pickle as pkl

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter


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
    yb = np.array([])
    yl = np.array([])
    Ks = np.array([])
    RTs = np.array([])
    if is_train:
      X, yb, yl, Ks, RTs, fnames, indices = load_directory_sunrgbd(path, train_test_split, is_train, categories, use_rgb)
    else:
      X, yb, yl, Ks, RTs, fnames, indices = load_directory_sunrgbd(path, train_test_split, is_train, categories, use_rgb)
      #X, Ks, RTs, fnames = load_test_directory_sunrgbd(path, X_npy_path)

    # np.save(X_npy_path, X)
    # np.save(yb_npy_path, yb)
    # np.save(yl_npy_path, yl)
    # print("finished saving")
    new_ds = True
  return X, yb, yl, new_ds, Ks, RTs, fnames, indices

def load_test_directory_sunrgbd(path, X_npy_path):


  # Loop through buildings
  areas = ['test']

  input_data = []

  Ks = []
  RTs = []
  fnames = []
  total_regions = 0

  # Loop through buildings
  if areas is None:
    areas = sorted(listdir(path))

  for area in areas:
    counter = 0
    print("\t\tLoading area {}...".format(area))
    for room in listdir(join(path, area)):
      if not room.endswith('.ply'):
        continue

      room_path = join(path, area, room[:-4])


      # Load point cloud

      input_pc = read_ply(room_path+".ply")
      K = np.load(room_path+"_k.npy")
      if len(K.shape) == 1:
        K = np.reshape(K, (3, 3))
      RT = np.load(room_path+"_rt.npy")
      if len(RT.shape) == 1:
        RT = np.reshape(RT, (3, 3))
      input_pc = input_pc["points"].as_matrix(columns=["x", "y", "z", "r", "g", "b"])

      fbbox = []
      flabel = []
      matches = 0
     
      input_data.append(input_pc)
      Ks.append(K)
      RTs.append(RT)
      fnames.append(room_path)
      counter += 1
      if counter % 100 == 0:
        print("\t\t\tLoaded {} rooms".format(counter))

    print("\t\tLoaded {} regions from area {}".format(counter, area))
    total_regions += counter

  input_data = np.array(input_data)
  Ks = np.array(Ks)
  RTs = np.array(RTs)
  fnames = np.array(fnames)

  print("finished casting to np array")


  return input_data, Ks, RTs, fnames


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


  if is_train:
    areas = ['train']
  else:
    areas = ['test']

  input_data = []
  bboxes = []
  labels = []
  Ks = []
  RTs = []
  fnames = []
  total_regions = 0
  index = 0
  indices = []
  ds = []

  # Loop through buildings
  if areas is None:
    areas = sorted(listdir(path))

  for area in areas:
    counter = 0
    print("\t\tLoading area {}...".format(area))
    for room in listdir(join(path, area)):
      if not room.endswith('.ply'):
        continue

      room_path = join(path, area, room[:-4])


      # Load point cloud
      categories = np.load(room_path+"_labels.npy")

      input_pc = read_ply(room_path+".ply")
      bbox = np.load(room_path+"_bboxes.npy")
      K = np.load(room_path+"_k.npy")
      if len(K.shape) == 1:
        K = np.reshape(K, (3, 3))
      RT = np.load(room_path+"_rt.npy")
      if len(RT.shape) == 1:
        RT = np.reshape(RT, (3, 3))
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
        Ks.append(K)
        RTs.append(RT)
        fnames.append(room_path)
        indices.append(index)
        counter += 1
        if counter % 100 == 0:
          print("\t\t\tLoaded {} rooms".format(counter))

        minx = min(input_pc[:,0])
        maxx = max(input_pc[:,0])
        x_d = maxx - minx
        miny = min(input_pc[:,1])
        maxy = max(input_pc[:,1])
        y_d = maxy - miny
        minz = min(input_pc[:,2])
        maxz = max(input_pc[:,2])
        z_d = maxz - minz
        if (max([x_d, y_d, z_d]) < 19):
          ds.append(max([x_d, y_d, z_d]))

      index += 1

    print("\t\tLoaded {} regions from area {}".format(counter, area))
    total_regions += counter

  input_data = np.array(input_data)
  bboxes = np.array(bboxes)
  labels = np.array(labels)
  Ks = np.array(Ks)
  RTs = np.array(RTs)
  fnames = np.array(fnames)
  # plt.hist(ds, bins = 50)
  # plt.show()

  print("finished casting to np array")


  return input_data, bboxes, labels, Ks, RTs, fnames, indices



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
    X, yb, yl, _, _, fnames, _ = load_directory_matterport(path, train_test_split, is_train, categories, use_rgb)
    np.save(X_npy_path, X)
    np.save(yb_npy_path, yb)
    np.save(yl_npy_path, yl)
    new_ds = True
  return X, yb, yl, new_ds, _, _, fnames, _

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
  ds = []
  input_data = []
  bboxes = []
  labels = []
  fnames = []
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
        fnames.append(room_path)
        # minx = min(input_pc[:,0])
        # maxx = max(input_pc[:,0])
        # x_d = maxx - minx
        # miny = min(input_pc[:,1])
        # maxy = max(input_pc[:,1])
        # y_d = maxy - miny
        # minz = min(input_pc[:,2])
        # maxz = max(input_pc[:,2])
        # z_d = maxz - minz
        # if (max([x_d, y_d, z_d]) < 19):
        #   ds.append(max([x_d, y_d, z_d]))

    print("\t\tLoaded {} regions from area {}".format(ri, area))
    total_regions += ri

  input_data = np.array(input_data)
  bboxes = np.array(bboxes)
  labels = np.array(labels)
  # plt.hist(ds, bins = 50)
  # plt.show()

  return input_data, bboxes, labels, None, None, fnames, None


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
    X, ys, yl, _, _, fnames, _ = load_directory_stanford(path, areas, categories)
    np.save(X_npy_path, X)
    np.save(ys_npy_path, ys)
    np.save(yl_npy_path, yl)
    new_ds = True
  return X, ys, yl, new_ds, None, None, fnames, None


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
  fnames = []
  ds = []
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
        # input_pc = np.genfromtxt(join(room_path, room+'.txt'), dtype=np.float32)
        # pc_df = pd.DataFrame()
        # pc_df['x'] = input_pc[:,0]
        # pc_df['y'] = input_pc[:,1]
        # pc_df['z'] = input_pc[:,2]
        # pc_df['r'] = input_pc[:,3]
        # pc_df['g'] = input_pc[:,4]
        # pc_df['b'] = input_pc[:,5]
        # write_ply(join(room_path, room+'.ply'), points=pc_df)


        input_pc = read_ply(join(room_path, room+'.ply'))
        input_pc = input_pc["points"].as_matrix(columns=["x", "y", "z", "r", "g", "b"])

        # minx = min(input_pc[:,0])
        # maxx = max(input_pc[:,0])
        # x_d = maxx - minx
        # miny = min(input_pc[:,1])
        # maxy = max(input_pc[:,1])
        # y_d = maxy - miny
        # minz = min(input_pc[:,2])
        # maxz = max(input_pc[:,2])
        # z_d = maxz - minz
        # if (max([x_d, y_d, z_d]) < 15):
        #   ds.append(max([x_d, y_d, z_d]))


        annotation_pc = np.array(annotation_pc)
        
        input_data.append(input_pc)
        segmentations.append(annotation_pc)
        labels.append(annotation_label)
        fnames.append(room_path)

  input_data = np.array(input_data)
  segmentations = np.array(segmentations)
  labels = np.array(labels)

  # plt.hist(ds, bins = 50)
  # plt.show()

  exit()

  return input_data, segmentations, labels, None, None, fnames, None


def load_npy(X_path, ys_path, yl_path):
  """
  Loads dataset from pre-loaded path if available.
  """
  assert exists(X_path), "Train npy file (X) does not exist."
  assert exists(ys_path), "Train npy file (ys) does not exist."
  assert exists(yl_path), "Train npy file (yl) does not exist."
  return np.load(X_path), np.load(ys_path), np.load(yl_path)