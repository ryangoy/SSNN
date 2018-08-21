import numpy as np
import shutil
import sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.misc import imread
from os.path import join, isdir
from os import makedirs
import sys
sys.path.append('..')
import pickle as pkl
from compute_mAP3 import compute_mAP
# /checkpoints/weights.17-2.62.hdf5 is best checkpoint

def combine_2d_3d(img, labels, labels_conf, preds, preds_conf, img_index, name="", threshold=0.5):
    
    good_preds = []
    for index in range(len(preds)):
        pred = preds[index]
        pred_conf = preds_conf[index]
        pred = np.array([min(pred[0], pred[3]), min(pred[1], pred[4]), max(pred[0], pred[3]), max(pred[1], pred[4])])
        for label, label_conf in zip(labels, labels_conf):
            label = np.array([min(label[0], label[2]), min(label[1], label[3]), max(label[0], label[2]), max(label[1], label[3])])
            max_LL = np.max(np.array([pred[:2], label[:2]]), axis=0)
            min_UR = np.min(np.array([pred[2:], label[2:]]), axis=0)
            intersection = max(0, np.prod(min_UR - max_LL))

            union = np.prod(pred[2:]-pred[:2]) + np.prod(label[2:]-label[:2]) - intersection

            # If we found a label that matches our prediction, i.e. a true positive
            if min(min_UR - max_LL) > 0 and intersection/union > threshold:
                preds_conf[index] = [label_conf*0.5 + 0.5*pred_conf[0]]
                good_preds.append(pred)
                break
            else:
                preds_conf[index] = [0.5*pred_conf[0]]

    plt.imshow(img / 255.)
    currentAxis = plt.gca()

    for bbox in labels:
        coords = (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='red', linewidth=2))

    for bbox, conf in zip(preds, preds_conf):
        if max(conf) > 0.8:
            coords = (bbox[0], bbox[1]), bbox[3]-bbox[0], bbox[4]-bbox[1]
            currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='blue', linewidth=2))

    for bbox in good_preds:
        coords = (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='green', linewidth=2))
    plt.savefig(join('figs', name+'_'+str(img_index) + '.png'))
    #plt.show()
    plt.clf()

    return preds_conf


def proj_3d(cls_3d, loc_3d, K, RT, fnames):

    R = RT[:, :3]
    #T = RT[:, 3:].T
    loc_3d = np.concatenate([loc_3d[:, :3]-loc_3d[:, 3:6], loc_3d[:, :3]+loc_3d[:, 3:6]], axis=-1)
    un_r_ll = np.dot(loc_3d[:, :3], R)
    un_r_ur = np.dot(loc_3d[:, 3:], R)

    un_rt_ll = un_r_ll# + T
    un_rt_ur = un_r_ur# + T

    un_rt_ll = np.array([un_rt_ll[:,0], -un_rt_ll[:,2], un_rt_ll[:,1]]).T
    un_rt_ur = np.array([un_rt_ur[:,0], -un_rt_ur[:,2], un_rt_ur[:,1]]).T

    ll_2d = np.dot(un_rt_ll, K.T)
    ur_2d = np.dot(un_rt_ur, K.T)

    ll_2d = ll_2d / np.tile(np.expand_dims(ll_2d[:,-1],-1), (1, 3))
    ur_2d = ur_2d / np.tile(np.expand_dims(ur_2d[:,-1],-1), (1, 3))

    proj_bboxes = np.concatenate([ll_2d, ur_2d], axis=-1)


    return proj_bboxes


def untransform_bboxes(preds, transforms):
    
    new_preds = []
    # new_labels = []
    index = 0

    for scene in range(len(preds)):
        t = transforms['t'][scene]
        s = transforms['s'][scene]
        scene_preds = []
        scene_labels = []
        for i in range(len(preds[scene])):
            pred = preds[scene][i]
            pred = np.concatenate([pred[:3]+t, pred[3:6], pred[6:]])
            scene_preds.append(pred)

        # for i in range(len(labels[scene])):
        #     label = labels[scene][i]
        #     scene_labels.append(label/mult_dims + bmins)

        new_preds.append(np.array(scene_preds))
        # new_labels.append(np.array(scene_labels))

    return np.array(new_preds)#, np.array(new_labels)


if __name__ == '__main__':
    if not isdir('figs'):
        makedirs('figs')
        #shutil.rmtree('figs')
    #makedirs('figs')
    class_name = sys.argv[2]
    class_mapping = {'bathtub': 0, 'bed': 1, 'bookshelf':2, 'chair':3, 'desk':4, 
                 'dresser':5, 'nightstand':6, 'night_stand':6, 'sofa':7, 'table':8, 'toilet':9}
    class_id = class_mapping[class_name]
    outputs_dir = sys.argv[1]
    transforms = pkl.load(open(join(outputs_dir,"test_transforms.pkl"), "rb"))


    bboxes_conf = np.load(join(outputs_dir, "bbox_cls_predictions.npy"))
    bboxes = np.load(join(outputs_dir, "bbox_predictions.npy"))
    labels = np.load('/home/ryan/cs/repos/ssd_keras/ssd_bboxes.npy', encoding="latin1")
    labels_conf = np.load('/home/ryan/cs/repos/ssd_keras/ssd_cls.npy', encoding="latin1")

    # if len(sys.argv) > 3:
    #     indices = np.load(join(outputs_dir, 'indices.npy'))
    #     print(indices)
    #     print(len(indices))
    #     labels = labels[indices]
    #     labels_conf = labels_conf[indices]

    class_bboxes = []
    for sc in range(len(labels)):
        scene_bboxes = []
        for obj in range(len(labels_conf[sc])):
            if labels_conf[sc][obj, class_id] > 0.5:
                # print(labels_conf[sc][obj].shape)
                # print(labels_conf[sc][obj, class_id])
                scene_bboxes.append(labels[sc][obj])
        class_bboxes.append(np.array(scene_bboxes))

    #labels = class_bboxes


    bboxes_cls = []
    bboxes = untransform_bboxes(bboxes, transforms)
    fnames = np.load(join(outputs_dir, "test_fnames.npy"))


    # for each scene...
    for index in range(len(labels)):
        fname = fnames[index]
        img = np.load(fname+'_rgb.npy')
        cls_3d = bboxes_conf[index]
        bboxes_3d = bboxes[index]
        K = np.load(fname+'_k.npy')
        if len(K.shape) == 1:
            K= K.reshape((3,3))
        RT = np.load(fname+'_rt.npy')
    
        #label_proj_bboxes = proj_3d(np.array(labels_conf[index]), np.array(labels[index]), K, RT, fname)
        if len(bboxes_3d) > 0 and len(labels_conf[index]) != 0:
            pred_proj_bboxes = proj_3d(cls_3d, bboxes_3d, K, RT, fname)
            # print(pred_proj_bboxes)
            g = combine_2d_3d(img, class_bboxes[index], labels_conf[index][:, class_id], pred_proj_bboxes, cls_3d, index, name=class_name)
            # print(g.shape)
            bboxes_cls.append(g)
        else:
            bboxes_cls.append(cls_3d)
    np.save(join(outputs_dir, 'bbox_cls_predictions_combined.npy'), np.array(bboxes_cls))
    #compute_mAP(bboxes, bboxes_conf, labels, labels_conf, threshold=0.25)
