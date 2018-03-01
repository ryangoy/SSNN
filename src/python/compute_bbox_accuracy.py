import numpy as np
import sys
import functools


def compute_accuracy(preds, labels):
    true_arr = []
    false_arr = []
    total_label_positives = 0
    total_found_positives = 0
    total_pred_positives = 0
    total_found_negatives = 0
    for scene in range(len(preds)):
        false_positives = 0
        labels_matched = np.zeros(len(labels[scene]))
        for pred in preds[scene]:
            pred_matched = False
            for i in range(len(labels[scene])):
                label = labels[scene][i]
                max_LL = np.max(np.array([pred[:3], label[:3]]), axis=0)
                min_UR = np.min(np.array([pred[3:], label[3:]]), axis=0)
                intersection = np.prod(min_UR - max_LL)
                union = np.prod(pred[3:]-pred[:3]) + np.prod(label[3:]-label[:3]) - intersection
                
                # we found a label that matches our prediction--a true positive
                if min(min_UR - max_LL) > 0 and intersection/union > 0.2:
                    if not pred_matched:
                        pred_matched = True
                        labels_matched[i] = 1

            # we couldn't find a single label to match our prediction -- a false positive
            if not pred_matched:
                false_positives += 1

        true_positives = sum(labels_matched)
        if len(labels[scene]) == 0:
            true_arr.append(1)
        else:
            # sometimes we may have multiple predictions map to the same ground truth box
            true_arr.append(min(true_positives/len(labels[scene]), 1))
        if len(preds[scene]) == 0:
            false_arr.append(0)
        else:
            false_arr.append((false_positives)/len(preds[scene]))
        total_label_positives += max(len(labels[scene]), true_positives)
        total_found_positives += true_positives
        total_pred_positives += len(preds[scene])
        total_found_negatives += false_positives
        print("{}/{} true positives for scene {}.".format(true_positives, max(len(labels[scene]), true_positives), scene))
        print("{}/{} false positives for scene {}.".format(false_positives, len(preds[scene]), scene))
    avg_true = functools.reduce(lambda x, y: x+y, true_arr)
    avg_false = functools.reduce(lambda x, y: x+y, false_arr)

    print("Average true positive ratio: {}".format(float(avg_true)/len(true_arr)))
    print("Average false positive ratio: {}".format(float(avg_false)/len(false_arr)))
    print("Overall true positive ratio: {}".format(float(total_found_positives)/total_label_positives))
    print("Overall false positive ratio: {}".format(float(total_found_negatives)/total_pred_positives))


if __name__ == '__main__':
    preds = np.load(sys.argv[1])
    labels = np.load(sys.argv[2])
    compute_accuracy(preds, labels)
