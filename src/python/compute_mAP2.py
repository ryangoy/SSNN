import numpy as np
import sys
import functools


# Retruns precision and recall arrays of a given sccene and category
def compute_PR_curve(sorted_preds, sorted_preds_confs, labels, labels_confs, threshold=0.5):
    curr_preds = [] 
    Ps = []
    Rs = []
    matched_labels = []
    if len(labels) == 0:
        Ps.append(1.0)
        Rs.append(1.0)
        return Ps, Rs

    # For no predictions
    Ps.append(1.0)
    Rs.append(0.0)

    # Loop through all the predictions for a scene in descending order of confidence values. Calculates AP.
    for pred in sorted_preds:

        # Find a label that the prediction corresponds to if it exists.
        pred_matched = False     
        for i in range(len(labels)):
            label = labels[i]
            if i in matched_labels:
                continue
            max_LL = np.max(np.array([pred[:3], label[:3]]), axis=0)
            min_UR = np.min(np.array([pred[3:], label[3:]]), axis=0)
            intersection = np.prod(min_UR - max_LL)
            union = np.prod(pred[3:]-pred[:3]) + np.prod(label[3:]-label[:3]) - intersection
            
            # If we found a label that matches our prediction, i.e. a true positive
            if min(min_UR - max_LL) > 0 and intersection/union > threshold and labels_confs[i] == 1:
                curr_preds.append(1)
                pred_matched = True
                matched_labels.append(i)
                break

        # If we couldn't find a single label to match our prediction, i.e. a false positive
        if not pred_matched:
            curr_preds.append(0)

        precision = float(sum(curr_preds)) / len(curr_preds)
        recall = float(sum(curr_preds)) / len(labels)
        Ps.append(precision)
        Rs.append(recall)

    return Ps, Rs

def compute_AP_from_PR(Ps, Rs):
    # Calculate AP for scene:
    PR_vals = []
    for recall_threshold in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        
        if len(Ps) == 0:
            PR_vals.append(0)
            continue

        i = np.argmax(Ps)

        while Rs[i] < recall_threshold:
            del Rs[i]
            del Ps[i]

            if len(Ps) == 0:
                break
            else:
                i = np.argmax(Ps)
        if len(Ps) == 0:
            PR_vals.append(0)
        else:
            PR_vals.append(Ps[i])
    assert len(PR_vals) == 11
    return PR_vals

def compute_mAP(preds, preds_conf, labels, labels_conf, hide_print=False):

    scene_APs = []


    # Loop through each scene. The mAP is calculated from the mean AP of each scene.
    for scene in range(len(preds)):

        # If there are no predictions (this should not happen if there is no conf threshold).
        if len(preds[scene]) == 0:
            if len(labels[scene]) != 0:
                scene_APs.append(0)
            else:
                scene_APs.append(1)
            continue

        # Fix this: for each category?
        category_APs = []
        for c in range(len(preds_conf[scene][0])):

            category_pred_confs = preds_conf[scene][:, c]
            category_labels_confs = np.array(labels_conf[scene])[:, c+1]
            category_labels = np.array(labels[scene])
            category_sorted_conf_indices = np.argsort(category_pred_confs)[::-1]
            category_sorted_preds = preds[scene][category_sorted_conf_indices]
            category_sorted_pred_confs = category_pred_confs[category_sorted_conf_indices]

            Ps, Rs = compute_PR_curve(category_sorted_preds, category_sorted_pred_confs, category_labels, category_labels_confs)
            PR_vals = compute_AP_from_PR(Ps, Rs)
            category_APs.append(sum(PR_vals)/11)

        scene_APs.append(sum(category_APs) / len(category_APs))
        
    mAP = sum(scene_APs) / len(scene_APs)

    if not hide_print:
        print("mAP is {}".format(mAP))


    return mAP

if __name__ == '__main__':
    preds = np.load(sys.argv[1])
    preds_conf = np.load(sys.argv[2])
    labels = np.load(sys.argv[3])
    labels_conf = np.load(sys.argv[4]) # tells us the category, not really the confidence
    mAP = compute_mAP(preds, preds_conf, labels, labels_conf)