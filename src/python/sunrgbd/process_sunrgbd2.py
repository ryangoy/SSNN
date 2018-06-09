import numpy as np
from scipy.misc import imread, imsave
from os import listdir, makedirs
from os.path import join, isdir, exists, basename
import json
import os
import pandas as pd
from shutil import rmtree
from pyntcloud.io import read_ply, write_ply
from scipy.io import loadmat

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
    # T = RT[:, 3:]
    Y = KiX
    # Y = KiX - np.tile(T, (1, KiX.shape[0])).T
    Y = Y.dot(np.linalg.inv(R))

    # add color to points
    colored_pc = np.concatenate([Y, rgb_img], axis=-1)

    return colored_pc


def process_sunrgbd(path):


    if isdir(join(path, 'train')):
        rmtree(join(path, 'train'))

    if isdir(join(path, 'val')):
        rmtree(join(path, 'val'))

    makedirs(join(path, 'train'))
    makedirs(join(path, 'val'))

    gt_data = loadmat(join(path, 'SUNRGBDMeta3DBB_v2.mat'))['SUNRGBDMeta'][0]
    tv_split = loadmat(join(path, 'allsplit.mat'))['trainvalsplit'][0,0]
    t_dirs = tv_split[0][:,0]
    v_dirs = tv_split[1][:,0]

    t_dirs = [str(t[0]) for t in t_dirs]
    v_dirs = [str(v[0]) for v in v_dirs]

    t_dirs.sort()
    v_dirs.sort()
    num_skipped = 0
    scene_index = 0
    for scene_gt in gt_data:
        scene_path = scene_gt[0][0]
        intrinsics = scene_gt[2]
        extrinsics = scene_gt[1]
        d_name = scene_gt[6][0]
        rgb_name = scene_gt[7][0]

        try:
            annotations = scene_gt[10][0]
        except:
            num_skipped += 1
            continue

        thetas = []
        coeffs = []
        centroids = []
        classnames = []
        for annotation in annotations:
            rot_mat = annotation[0]
            if len(rot_mat.shape) == 1:
                rot_mat = np.reshape(rot_mat, (3,3))
            thetas.append(np.arccos(rot_mat[0,0]))
            coeffs.append(annotation[1][0])
            centroids.append(annotation[2][0])
            classnames.append(annotation[3][0])


        bbox_loc = np.concatenate([np.array(centroids), np.array(coeffs), np.array(thetas).reshape(-1,1)], axis=-1)
        bbox_cls = np.array(classnames)
        

        # if join('/n/fs/sun3d/data/', scene_path + '/') in t_dirs:
        #     save_path = join(path, 'train', str(rgb_name)[:-4])
        # elif join('/n/fs/sun3d/data/', scene_path + '/') in v_dirs:
        #     save_path = join(path, 'val', str(rgb_name)[:-4])
        # else:
        #     # print("{} not in train or val".format(join('/n/fs/sun3d/data/', scene_path)))
        #     num_skipped += 1
        #     continue

        save_path = join(path, 'train', basename(str(scene_path)))

        rgb_img = imread(join(path, scene_path[8:], 'image', str(rgb_name)))
        d_img = imread(join(path, scene_path[8:], 'depth', str(d_name)))
        colored_pc = rgbd2pc(rgb_img, d_img, intrinsics, extrinsics).astype('float32')
        result = pd.DataFrame(dtype='float32')
        result["x"] = colored_pc[:,0]
        result["y"] = colored_pc[:,1]
        result["z"] = colored_pc[:,2]

        result["r"] = colored_pc[:,3]
        result["g"] = colored_pc[:,4]
        result["b"] = colored_pc[:,5]

        # bbox_pcs = []
        # bbox_loc = []
        # bbox_cls = []
        write_ply(save_path+'.ply', points=result)
        np.save(save_path+'_rgb.npy', rgb_img)
        np.save(save_path+'_k.npy', intrinsics)
        np.save(save_path+'_d.npy', d_img)
        np.save(save_path+'_rt.npy', extrinsics)
        np.save(save_path+'_bboxes.npy', np.array(bbox_loc))
        np.save(save_path+'_labels.npy', np.array(bbox_cls))

        scene_index += 1

        if scene_index % 100 == 0:
            print('\tProcessed {}/{}.'.format(scene_index, len(gt_data)))

    print('Skipped {} entries'.format(num_skipped))


def process_test_sunrgbd(path):


    if isdir(join(path, 'test')):
        rmtree(join(path, 'test'))

    makedirs(join(path, 'test'))

    gt_data = loadmat(join(path, 'SUNRGBDMetaStructIOTest.mat'))['SUNRGBDMetaStructIOTest'][0]
    tv_split = loadmat(join(path, 'allsplit.mat'))['trainvalsplit'][0,0]
    t_dirs = tv_split[0][:,0]
    v_dirs = tv_split[1][:,0]

    t_dirs = [str(t[0]) for t in t_dirs]
    v_dirs = [str(v[0]) for v in v_dirs]

    t_dirs.sort()
    v_dirs.sort()
    num_skipped = 0
    scene_index = 0
    for scene_gt in gt_data:
        scene_path = scene_gt[0][0]
        intrinsics = scene_gt[3]
        extrinsics = scene_gt[2]
        d_name = scene_gt[7][0]
        rgb_name = scene_gt[8][0]

        save_path = join(path, 'test', basename(str(scene_path)))

        rgb_img = imread(join(path, "sunrgbd_test", scene_path[14:], 'image', str(rgb_name)))
        d_img = imread(join(path, "sunrgbd_test", scene_path[14:], 'depth', str(d_name)))
        colored_pc = rgbd2pc(rgb_img, d_img, intrinsics, extrinsics).astype('float32')
        result = pd.DataFrame(dtype='float32')
        result["x"] = colored_pc[:,0]
        result["y"] = colored_pc[:,1]
        result["z"] = colored_pc[:,2]

        result["r"] = colored_pc[:,3]
        result["g"] = colored_pc[:,4]
        result["b"] = colored_pc[:,5]

        write_ply(save_path+'.ply', points=result)
        np.save(save_path+'_rgb.npy', rgb_img)
        np.save(save_path+'_k.npy', intrinsics)
        np.save(save_path+'_d.npy', d_img)
        np.save(save_path+'_rt.npy', extrinsics)
        scene_index += 1

        if scene_index % 100 == 0:
            print('\tProcessed {}/{}.'.format(scene_index, len(gt_data)))

    print('Skipped {} entries'.format(num_skipped))

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
    process_sunrgbd('/media/ryan/sandisk/SUNRGBD/')
