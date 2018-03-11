import math
import numpy as np

def compute_map(preds, cls_vals, labels, categories):
    recall_range = np.linspace(0, 1, 11)
    aps = np.zeros(len(categories))
    # for each category 
    for i in range(len(categories)):
        num_labels_matched = 0
        correct = 0
        incorrect = 0
        best_precisions = np.zeros(11)
        preds_class, conf_class = bboxes_by_class(preds, cls_vals, categories[i])
        labels_class = labels_by_class(labels)
        num_labels = count_labels(labels_class)
        matched_labels = set()
        flattened_conf, conf_indices = flatten_confs(conf_class)
        sorted_indices = sorted(conf_indices, key=lambda x: conf_class[x[0]][x[1]])
        for j in range(len(sorted_indices))
            room_idx = sorted_indices[j][0]
            for k in range(len(labels_class[room_idx])):
                # TODO compute overlap
                
                if overlap > 0.5:
                    correct += 1
                    if (room_idx, k) not in matched_labels: 
                        num_labels_matched += 1
                        match_labels.add((room_idx, k))
                else:
                    incorrect += 1
                curr_recall = num_labels_matched / num_labels
                curr_precision = correct / (correct + incorrect)
                
                for rr in range(len(recall_range)):
                    if curr_recall < recall_range[rr]:
                        break
                    if curr_precision > best_precisions[rr]:
                        best_precisions[rr] = curr_precision 
        ap = np.mean(best_precisions)
        print('category', categories[i], 'ap', ap)
        aps[i] = ap

    print('mAP', np.mean(aps))        

def count_labels(labels):
    s = 0
    for i n range(len(labels)):
        for j in range(len(labels[i])):
            s += 1
    return s

def label_by_class(labels):
    #TODO

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
    for i in range(len(preds)):
        preds_room = []
        cls_room = []
        for j in range(len(preds[i])):
            if cls_vals[i][j].argmax() == category:
                preds_room.append(preds[i][j])
                cls_room.append(cls_vals[i][j][category])
        resulting_preds.append(preds_room)
        resulting_conf.append(cls_room)
    return resulting_preds, resulting_conf

if __name__=='main':
    preds = np.load(sys.argv[1])
    cls_vals = np.load(sys.argv[2])
    labels = np.load(sys.argv[3])
    categories = np.load(sys.argv[4])
    compute_map(preds, cls_vals, labels, categories)

