import numpy as np
import sys
import matplotlib.pyplot as plt
from scipy.misc import imread
from os.path import join
import sys
sys.path.append('..')
import pickle as pkl
from compute_mAP3 import compute_mAP

# /checkpoints/weights.17-2.62.hdf5 is best checkpoint

def combine_2d_3d(img, labels, labels_conf, preds, preds_conf, threshold=0.5):
    
    good_preds = []
    for index in range(len(preds)):
        pred = preds[index]
        pred_conf = preds_conf[index]
        pred = np.array([min(pred[0], pred[3]), min(pred[1], pred[4]), max(pred[0], pred[3]), max(pred[1], pred[4])])
        for label, label_conf in zip(labels, labels_conf):
            label = np.array([min(label[0], label[3]), min(label[1], label[4]), max(label[0], label[3]), max(label[1], label[4])])
            max_LL = np.max(np.array([pred[:2], label[:2]]), axis=0)
            min_UR = np.min(np.array([pred[2:], label[2:]]), axis=0)
            intersection = max(0, np.prod(min_UR - max_LL))

            union = np.prod(pred[2:]-pred[:2]) + np.prod(label[2:]-label[:2]) - intersection

            # If we found a label that matches our prediction, i.e. a true positive
            if min(min_UR - max_LL) > 0 and intersection/union > threshold:
                print("Found something")
                print(pred_conf)
                preds_conf[index] = [(1.0-pred_conf[0])/2 + pred_conf[0]]

                good_preds.append(pred)
                break
            else:
                preds_conf[index] = [pred_conf[0]/2]

    plt.imshow(img / 255.)
    currentAxis = plt.gca()
    for bbox in labels:
        coords = (bbox[0], bbox[1]), bbox[3]-bbox[0], bbox[4]-bbox[1]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='red', linewidth=2))

    for bbox in preds:
        coords = (bbox[0], bbox[1]), bbox[3]-bbox[0], bbox[4]-bbox[1]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='blue', linewidth=2))

    for bbox in good_preds:
        coords = (bbox[0], bbox[1]), bbox[2]-bbox[0], bbox[3]-bbox[1]
        currentAxis.add_patch(plt.Rectangle(*coords, fill=False, edgecolor='green', linewidth=2))

    plt.show()

    # return label_conf


def proj_3d(cls_3d, loc_3d, K, RT, fnames):

    R = RT[:, :3]
    T = RT[:, 3:].T

    un_r_ll = np.dot(loc_3d[:, :3], R)

    un_r_ur = np.dot(loc_3d[:, 3:], R)

    un_rt_ll = un_r_ll + T
    un_rt_ur = un_r_ur + T

    un_rt_ll = np.array([un_rt_ll[:,0], -un_rt_ll[:,2], un_rt_ll[:,1]]).T
    un_rt_ur = np.array([un_rt_ur[:,0], -un_rt_ur[:,2], un_rt_ur[:,1]]).T

    ll_2d = np.dot(un_rt_ll, K.T)
    ur_2d = np.dot(un_rt_ur, K.T)

    ll_2d = ll_2d / np.tile(np.expand_dims(ll_2d[:,-1],-1), (1, 3))
    ur_2d = ur_2d / np.tile(np.expand_dims(ur_2d[:,-1],-1), (1, 3))

    proj_bboxes = np.concatenate([ll_2d, ur_2d], axis=-1)


    return proj_bboxes


def untransform_bboxes(preds, labels, transforms):
    
    new_preds = []
    new_labels = []
    index = 0

    for scene in range(len(preds)):
        t = transforms['t'][scene]
        s = transforms['s'][scene]
        scene_preds = []
        scene_labels = []
        mult_dims = np.array([s[0], s[1], s[2], s[0], s[1], s[2]])
        bmins = np.array([t[0], t[1], t[2], t[0], t[1], t[2]])
        for i in range(len(preds[scene])):#zip(preds[scene], labels[scene]):
            pred = preds[scene][i]
            scene_preds.append(pred/mult_dims + bmins)
            

        for i in range(len(labels[scene])):
            label = labels[scene][i]
            scene_labels.append(label/mult_dims + bmins)


        new_preds.append(np.array(scene_preds))
        new_labels.append(np.array(scene_labels))

    return np.array(new_preds), np.array(new_labels)




if __name__ == '__main__':

    transforms = pkl.load(open("../test_transforms.pkl", "rb"))

    outputs_dir = sys.argv[1]
    bboxes_conf = np.load(join(outputs_dir, "bbox_cls_predictions.npy"))
    bboxes = np.load(join(outputs_dir, "bbox_predictions.npy"))

    labels = np.load(join(outputs_dir, "bbox_test_labels.npy"))
    labels_conf = np.load(join(outputs_dir, "bbox_test_cls_labels.npy"))

    bboxes_cls = []

    bboxes, labels = untransform_bboxes(bboxes, labels, transforms)


    for index in range(10):
        fname = np.load(sys.argv[2])[index]
        print(fname)
        img = np.load(fname+'_rgb.npy')
        cls_3d = bboxes_conf[index]
        bboxes_3d = bboxes[index]
        K = np.load(fname+'_k.npy')
        if len(K.shape) == 1:
            K= K.reshape((3,3))
        RT = np.load(fname+'_rt.npy')
    
        label_proj_bboxes = proj_3d(np.array(labels_conf[index]), np.array(labels[index]), K, RT, fname)
        pred_proj_bboxes = proj_3d(cls_3d, bboxes_3d, K, RT, fname)
        bboxes_cls.append(combine_2d_3d(img, label_proj_bboxes, labels_conf[index], pred_proj_bboxes, cls_3d))

    np.save('toilet_bbox.npy', bboxes)
    compute_mAP(bboxes, bboxes_conf, labels, labels_conf, threshold=0.25)