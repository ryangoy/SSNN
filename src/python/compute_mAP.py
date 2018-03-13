import sys
import math
import numpy as np

def count_labels(labels):
    s = 0
    for i in range(len(labels)):
        for j in range(len(labels[i])):
            s += 1
    return s

def flatten_confs(conf_class):
    flattened_list = []
    indices = []
    for i in range(len(conf_class)):
        for j in range(len(conf_class[i])):
            flattened_list.append(conf_class[i][j])
            indices.append((i, j))
    return flattened_list, indices

def bboxes_by_class(preds, cls_vals, category):
    resulting_preds = []
    resulting_conf = []

    # for each room
    for i in range(len(cls_vals)):
        preds_room = []
        cls_room = []
        for j in range(len(cls_vals[i])):
            if cls_vals[i][j].argmax() == category:
                preds_room.append(preds[i][j])
                cls_room.append(cls_vals[i][j][category])
        resulting_preds.append(preds_room)
        resulting_conf.append(cls_room)
    return resulting_preds, resulting_conf

def compute_map(preds, cls_vals, loc_labels, cls_labels):
#    print(cls_vals)
    num_categories = len(cls_vals[1][0])
    recall_range = np.linspace(0, 1, 11)
    aps = np.zeros(num_categories)
    flattened_conf, conf_indices = flatten_confs(cls_vals)    
    # ignore negative class
    for i in range(1, num_categories+1):
        num_labels_matched = 0
        correct = 0
        incorrect = 0
        best_precisions = np.zeros(11)
        labels_class, _ = bboxes_by_class(loc_labels, cls_labels, i)
        num_labels = count_labels(labels_class)
        matched_labels = set()

        #rank predictions by their confidence in the current class
        sorted_indices = sorted(conf_indices, key=lambda x: cls_vals[x[0]][x[1]][i-1], reverse=True)
        for j in range(len(sorted_indices)):
            room_idx = sorted_indices[j][0]
            pred_bbox_idx = sorted_indices[j][1]
            box_matched = False
            pred = preds[room_idx][pred_bbox_idx]
            for k in range(len(labels_class[room_idx])):
                label = labels_class[room_idx][k]
                max_LL = np.max(np.array([pred[:3], label[:3]]), axis=0)
                min_UR = np.min(np.array([pred[3:], label[3:]]), axis=0)
                intersection = np.prod(min_UR - max_LL)
                union = np.prod(pred[3:]-pred[:3]) + np.prod(label[3:]-label[:3]) - intersection

                # we found a label that matches our prediction--a true positive
                if min(min_UR - max_LL) > 0 and intersection/union > 0.5:
                    correct += 1
                    box_matched = True
                    if (room_idx, k) not in matched_labels: 
                        num_labels_matched += 1
                        matched_labels.add((room_idx, k))
                    break
            
            if not box_matched:
                incorrect += 1
            curr_recall = num_labels_matched / num_labels
            curr_precision = correct / (correct + incorrect)
                
            for rr in range(len(recall_range)):
                if curr_recall < recall_range[rr]:
                    break
                if curr_precision > best_precisions[rr]:
                    best_precisions[rr] = curr_precision 

        print(best_precisions)    
        ap = np.mean(best_precisions)
        print('category', i, 'ap', ap)
        aps[i-1] = ap

    print('mAP', np.mean(aps))        
    return np.mean(aps)

if __name__=='__main__':
    preds = np.load(sys.argv[1])
    cls_vals = np.load(sys.argv[2])
    loc_labels = np.load(sys.argv[3])
    cls_labels = np.load(sys.argv[4])
    compute_map(preds, cls_vals, loc_labels, cls_labels)



