import numpy as np
from scipy.misc import imread, imsave
from os import listdir
from os.path import join, isdir, exists
import json
import os
import pandas as pd
from shutil import rmtree
from pyntcloud.io import read_ply, write_ply

def rgbd2pc(rgb_img, d_img, K, RT, KiX=None):

    index_matrix = create_index_matrix(rgb_img.shape[0], rgb_img.shape[1])

    # processing of depth map
    d_img = np.right_shift(d_img, 3)
    d_img = d_img.astype(float).flatten()/1000
    # d_img[d_img>8]=8
    rgb_img = rgb_img.reshape((-1, 3))


    # apply inverse K matrix
    if len(K.shape) == 1:
        K = K.reshape((3,3))

    cx = K[0,2]
    cy = K[1,2]
    fx = K[0,0]
    fy = K[1,1]

    x = (index_matrix[...,0]-cx) * d_img / fx
    y = (index_matrix[...,1]-cy) * d_img / fy
    z = d_img

    # join channels
    x = np.expand_dims(x, -1)
    y = np.expand_dims(y, -1)
    z = np.expand_dims(z, -1)
    KiX = np.concatenate([x, z, -y], axis=-1)

    # apply rotation and translation from extrinsics
    R = RT[:, :3]
    T = RT[:, 3:]
    Y = KiX - np.tile(T, (1, KiX.shape[0])).T
    Y = Y.dot(np.linalg.inv(R))

    # add color to points
    colored_pc = np.concatenate([Y, rgb_img], axis=-1)

    return colored_pc


def process_folder(data_path, save_path, fullres=False):

    if exists(save_path):
        rmtree(save_path)
    os.makedirs(save_path)

    scene_index = 0

    imgs = listdir(data_path)
    imgs = sorted(imgs)
    for img in imgs:
        folder_path = join(data_path, img)
        if isdir(folder_path):
            
            try:
                # extrinsics
                extrinsics_folder = join(folder_path, 'extrinsics')

                # sometimes there is more than 1 extrinsics file.
                extrinsics_file = join(extrinsics_folder, listdir(extrinsics_folder)[-1])
                extrinsics_npy = np.loadtxt(extrinsics_file)
                anno_extrinsics = extrinsics_npy[:3, :3]

                if fullres:
                    fullres_folder = join(folder_path, 'fullres')
                    if not exists(fullres_folder):
                        continue
                    rgb_img = None
                    d_img = None
                    intrinsics_npy = None
                    for f in listdir(fullres_folder):
                        if f.endswith('.jpg'):
                            rgb_img = imread(join(fullres_folder, f))
                        elif f.endswith('.png'):
                            d_img = imread(join(fullres_folder, f))
                        elif f.endswith('.txt'):
                            intrinsics_npy = np.loadtxt(join(fullres_folder, f))

                else:

                    intrinsics_npy = np.loadtxt(join(folder_path, 'intrinsics.txt'))
                    for f in listdir(join(folder_path, 'image')):
                        rgb_img = imread(join(folder_path, 'image', f))

                    for f in listdir(join(folder_path, 'depth_bfx')):
                        d_img = imread(join(folder_path, 'depth_bfx', f))

                    if rgb_img is None or d_img is None or intrinsics_npy is None:
                        print('Image didn\'t load in {}.'.format(folder_path))
                        continue
            
                raw_annotations = json.load(open(join(folder_path, 'annotation3Dfinal', 'index.json')))['objects']
            except FileNotFoundError:
                print("\tFolder {} was skipped due to missing information.".format(folder_path))
                continue

            colored_pc = rgbd2pc(rgb_img, d_img, intrinsics_npy, extrinsics_npy).astype('float32')
            result = pd.DataFrame(dtype='float32')
            result["x"] = colored_pc[:,0]
            result["y"] = colored_pc[:,1]
            result["z"] = colored_pc[:,2]

            result["r"] = colored_pc[:,3]
            result["g"] = colored_pc[:,4]
            result["b"] = colored_pc[:,5]

            # print(read_ply(join(save_path, 'region'+str(scene_index)+'.ply'))["points"].dtypes)

            bbox_pcs = []
            bbox_loc = []
            bbox_cls = []
            for raw_annot in raw_annotations:
                if raw_annot is None or type(raw_annot) is not dict:
                    continue
                for poly in raw_annot['polygon']:
                    bbox = annotation_to_bbox(poly, anno_extrinsics)
                    #bbox_pcs.append(bbox_to_pc(bbox))
                    bbox_loc.append(bbox)
                    bbox_cls.append(raw_annot['name'])
            # all_bbox_pcs = np.concatenate(bbox_pcs, axis=0)
            # bbox_color = np.zeros((all_bbox_pcs.shape))
            # bbox_color[:, 0] = 102
            # bbox_color[:, 1] = 255
            # bbox_color[:, 2] = 102
            # all_bbox_pcs = np.concatenate([all_bbox_pcs, bbox_color], axis=-1)
            if len(bbox_loc) > 0 and len(bbox_cls) > 0:
                write_ply(join(save_path, 'region'+str(scene_index)+'.ply'), points=result)
                np.save(join(save_path, 'region{}_bboxes.npy'.format(scene_index)), np.array(bbox_loc))
                np.save(join(save_path, 'region{}_labels.npy'.format(scene_index)), np.array(bbox_cls))
                
            else:
                print("\tFolder {} was skipped due to missing information.".format(folder_path))

            scene_index += 1

            
        if scene_index % 100 == 0:
            print('\tProcessed {}/{} scenes from {}.'.format(scene_index, len(imgs), data_path))

def process_sunrgbd(path):

    for sensor in ['xtion']:
    #for sensor in listdir(path):
        if sensor != 'SUNRGBDtoolbox':
            for dataset in listdir(join(path, sensor)):
                if not dataset.endswith('_processed') and isdir(join(path, sensor, dataset)):
                    print('Processing data from {}...'.format(join(path, sensor, dataset)))
                    process_folder(join(path, sensor, dataset), join(path, sensor, dataset+'_processed'))

def annotation_to_bbox(annotation, R):
    Xs = annotation['X']
    Zs = annotation['Z']
    Ymin = annotation['Ymin']
    Ymax = annotation['Ymax']

    xc = np.concatenate([Xs, Xs], axis=0)
    yc = np.array([Ymin]*len(Xs)+[Ymax]*len(Xs))
    zc = np.concatenate([Zs, Zs], axis=0)

    coords = np.stack([xc, yc, zc])

    # for some reason we don't have to apply any transformations here
    tcoords = coords

    tcoords = np.array([tcoords[0], tcoords[2], -tcoords[1]])
    bbox = np.concatenate([np.min(tcoords, axis=1), np.max(tcoords, axis=1)])

    return bbox

def bbox_to_pc(bbox):
    Xs = [0, 3]
    Ys = [1, 4]
    Zs = [2, 5]
    corners = []
    for X in Xs:
        for Y in Ys:
            for Z in Zs:
                corners.append([bbox[X], bbox[Y], bbox[Z]])

    points = []
    for c1 in corners:
        for c2 in corners:
            if c1 is c2:
                continue

            interval = np.tile(np.reshape(np.linspace(0, 1, num=300), (-1,1)), (1, 3))
            edge = (1-interval) * c1 + interval * c2
            points.append(edge)

    points = np.concatenate(points, axis=0)

    return points

def create_index_matrix(width, height):

    return np.stack(np.meshgrid(np.arange(1, height+1), np.arange(1, width+1)), axis=-1).reshape((-1, 2))

if __name__ == '__main__':
    process_sunrgbd('/home/ryan/cs/datasets/SUNRGBD/')
