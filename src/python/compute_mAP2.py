import numpy as np
import sys
import functools


# Retruns precision and recall arrays of a given sccene and category
def compute_PR_curve(sorted_preds, sorted_pred_scene_ids, labels, labels_confs, category, threshold=0.25):
    curr_preds = [] 
    Ps = [1.0, 0.0]
    Rs = [0.0, 1.0]
    matched_labels = []


    # Loop through all the predictions for a scene in descending order of confidence values. Calculates AP.
    for pred, scene_id in zip(sorted_preds, sorted_pred_scene_ids):

        # Find a label that the prediction corresponds to if it exists.
        pred_matched = False     
        for i in range(len(labels[scene_id])):
            label = labels[scene_id][i]
            if i in matched_labels:
                continue
            max_LL = np.max(np.array([pred[:3], label[:3]]), axis=0)
            min_UR = np.min(np.array([pred[3:], label[3:]]), axis=0)
            intersection = np.prod(min_UR - max_LL)
            union = np.prod(pred[3:]-pred[:3]) + np.prod(label[3:]-label[:3]) - intersection
            
            # If we found a label that matches our prediction, i.e. a true positive

            if min(min_UR - max_LL) > 0 and intersection/union > threshold and labels_confs[scene_id][i][category+1] == 1:
                print("hi")
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

    APs = []
    for c in range(len(preds_conf[0][0])):
        pred_scene_ids = []
        concat_preds = np.concatenate(preds)
        concat_preds_conf = np.concatenate(preds_conf)
        for scene in range(len(preds)):
            pred_scene_ids += [scene] * len(preds[scene])


        category_pred_confs = concat_preds_conf[:, c]
        category_sorted_conf_indices = np.argsort(category_pred_confs)[::-1]
        category_sorted_preds = concat_preds[category_sorted_conf_indices]
        sorted_pred_scene_ids = np.array(pred_scene_ids)[category_sorted_conf_indices]

        Ps, Rs = compute_PR_curve(category_sorted_preds, sorted_pred_scene_ids, labels, labels_conf, c)
        PR_vals = compute_AP_from_PR(Ps, Rs)

        APs.append(sum(PR_vals)/11)
        
    mAP = sum(APs) / len(APs)

    if not hide_print:
        print("mAP is {}".format(mAP))

    return mAP

if __name__ == '__main__':
    preds = np.load(sys.argv[1])
    preds_conf = np.load(sys.argv[2])
    labels = np.load(sys.argv[3])
    labels_conf = np.load(sys.argv[4]) # tells us the category, not really the confidence
    mAP = compute_mAP(preds, preds_conf, labels, labels_conf)