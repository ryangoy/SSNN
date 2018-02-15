import numpy as np
import sys
import functools


def compute_accuracy(preds, labels):
    true_arr = []
    false_arr = []
    for scene in range(len(preds)):
        true_positives = 0
        false_positives = 0
        for label in labels[scene]:
            label_matched = False
            for pred in preds[scene]:
            
                # pred_bbox = np.concatenate([pred[:3]-pred[3:]-1, pred[:3]+pred[3:]+1])
                # label_bbox = np.concatenate([label[:3]-label[3:]-1, label[:3]+label[3:]+1])

                max_LL = np.max(np.array([pred[:3], label[:3]]), axis=0)
                min_UR = np.min(np.array([pred[3:], label[3:]]), axis=0)
                intersection = np.prod(min_UR - max_LL)
                union = np.prod(pred[3:]-pred[:3]) + np.prod(label[3:]-label[:3]) - intersection
                # print 'stats'
                # print pred.astype(int)
                # print label.astype(int)
                # print intersection
                # print union
                if min(min_UR - max_LL) > 0 and intersection/union >0.25:
                    if not label_matched:
                        true_positives += 1
                        label_matched = True
                else:
                    false_positives += 1

        if len(labels[scene]) > 0:
            true_arr.append(float(true_positives)/len(labels[scene]))
        else:
            true_arr.append(1)
        if len(preds[scene]) > 0:
            #false_arr.append(float(len(preds[scene])-true_positives)/len(preds[scene]))
            false_arr.append(float(false_positives)/len(preds[scene]))
        else:
            false_arr.append(0)
        print("{}/{} true positives for scene {}.".format(true_positives, len(labels[scene]), scene))
        print("{}/{} false positives for scene {}.".format(len(preds[scene])-true_positives, len(preds[scene]), scene))
    avg_true = functools.reduce(lambda x, y: x+y, true_arr)
    avg_false = functools.reduce(lambda x, y: x+y, false_arr)

    print("Average true positive ratio: {}".format(float(avg_true)/len(true_arr)))
    print("Average false positive ratio: {}".format(float(avg_false)/len(false_arr)))
                


if __name__ == '__main__':
    preds = np.load(sys.argv[1])
    labels = np.load(sys.argv[2])
    compute_accuracy(preds, labels)
