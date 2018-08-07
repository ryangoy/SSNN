import numpy as np
from pyntcloud.io import read_ply, write_ply  
import sys
import pandas as pd


def bbox2pc(read_path, write_path, confidences=None):
    

    # shape: (num_bboxes, 6)
    bbox_npy = np.load(read_path)[1]

    bbox_pc = []

    if confidences is not None:
        confidences = np.load(confidences)
        print(confidences[1])
        conf_indices = np.argsort(confidences[1])[::-1]
        print(conf_indices)
        conf_indices = conf_indices[:5]
        bbox_npy = bbox_npy[conf_indices]
        

    for bbox in bbox_npy:
        bbox_pc.append(create_pc(bbox))

    bbox_pc = np.concatenate(bbox_pc, axis=0)
    bbox_color = np.zeros((bbox_pc.shape))
    bbox_color[:, 0] = 102
    bbox_color[:, 1] = 255
    bbox_color[:, 2] = 102
    bbox_pc = np.concatenate([bbox_pc, bbox_color], axis=-1)

    df = pd.DataFrame()

    df['x'] = bbox_pc[:, 0]
    df['y'] = bbox_pc[:, 1]
    df['z'] = bbox_pc[:, 2]
    df['r'] = bbox_pc[:, 3]
    df['g'] = bbox_pc[:, 4]
    df['b'] = bbox_pc[:, 5]

    # write_ply(write_path, points=df)
    np.savetxt(write_path, bbox_pc)


def create_pc(bbox_raw):

    bbox = np.concatenate([-bbox_raw[3:6], +bbox_raw[3:6]])
    sinval = np.sin(bbox_raw[6])
    cosval = np.cos(bbox_raw[6])

    R = np.array([[cosval, -sinval, 0], [sinval, cosval, 0], [0, 0, 1]])

    Xs = [0, 3]
    Ys = [1, 4]
    Zs = [2, 5]
    corners = []
    for X in Xs:
        for Y in Ys:
            for Z in Zs:
                corners.append([bbox[X], bbox[Y], bbox[Z]])

    points = []
    c1s = np.array([corners[0], corners[0], corners[0], corners[3], corners[3], corners[3], corners[5], corners[5], corners[5], corners[6], corners[6], corners[6]])
    c2s = np.array([corners[1], corners[2], corners[4], corners[7], corners[1], corners[2], corners[1], corners[4], corners[7], corners[2], corners[4], corners[7]])
    c1s = np.dot(c1s, R.T) + bbox_raw[:3]
    c2s = np.dot(c2s, R.T) + bbox_raw[:3]
    for c1, c2 in zip(c1s, c2s):

        interval = np.tile(np.reshape(np.linspace(0, 1, num=300), (-1,1)), (1, 3))
        edge = (1-interval) * c1 + interval * c2
        points.append(edge)

    points = np.concatenate(points, axis=0)

    return points


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage:\n\tpython bbox2pc.py <path to bbox npy file> <path to bbox save file> <path to bbox confidences (optional)>\n")
        exit()
    if len(sys.argv) == 3:
        bbox2pc(sys.argv[1], sys.argv[2])
    else:
        bbox2pc(sys.argv[1], sys.argv[2], sys.argv[3])