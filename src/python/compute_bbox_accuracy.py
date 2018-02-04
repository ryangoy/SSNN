import numpy as np
import sys

def compute_accuracy(preds, labels):
    true_arr = []
    false_arr = []
    for scene in range(len(preds)):
        true_positives = 0
        for pred in preds[scene]:
            for label in labels[scene]:
                pred_bbox = np.concatenate([pred[:3]-pred[3:]-1, pred[:3]+pred[3:]+1])
                label_bbox = np.concatenate([label[:3]-label[3:]-1, label[:3]+label[3:]+1])
                w = np.maximum(0, np.prod(np.abs(pred_bbox-label_bbox)))
                if w / np.prod(label_bbox[3:]-label_bbox[:3]) > 0.1:
                    true_positives += 1
                    break
        if len(labels[scene]) > 0:
            true_arr.append(float(true_positives)/len(labels[scene]))
        else:
            true_arr.append(1)
        if len(preds[scene]) > 0:
            false_arr.append(float(len(preds[scene])-true_positives)/len(preds[scene]))
        else:
            false_arr.append(0)
        print("{}/{} true positives for scene {}.".format(true_positives, len(labels[scene]), scene))
        print("{}/{} false positives for scene {}.".format(len(preds[scene])-true_positives, len(preds[scene]), scene))
    avg_true = reduce(lambda x, y: x+y, true_arr)
    avg_false = reduce(lambda x, y: x+y, false_arr)

    print("Average true positive ratio: {}".format(float(avg_true)/len(true_arr)))
    print("Average false positive ratio: {}".format(float(avg_false)/len(false_arr)))
                


if __name__ == '__main__':
    preds = np.load(sys.argv[1])
    labels = np.load(sys.argv[2])
    compute_accuracy(preds, labels)
