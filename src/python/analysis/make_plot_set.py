import numpy as np
import sys
sys.path.append('/home/sahinera/Documents/SSNN/SSNN/src/python/')
import utils

BASE_DIR = '/home/sahinera/Documents/SSNN/Stanford3dDataset_v1.2/outputs/'

bbox_labels = np.load(BASE_DIR + 'bbox_test_labels.npy')
bbox_cls_labels = np.load(BASE_DIR + 'bbox_test_cls_labels.npy')
bbox_preds = np.load(BASE_DIR + 'bbox_predictions.npy')
bbox_cls_preds = np.load(BASE_DIR + 'bbox_cls_predictions.npy')

relevant_room_list = []
bbox_table = []

for i in range(len(bbox_labels)):
    room_bbox = []
    for j in range(len(bbox_labels[i])):
        if bbox_cls_labels[i][j][0] == 1:
            relevant_room_list.append(i)
            room_bbox.append(bbox_labels[i][j])
    bbox_table.append(room_bbox)

bbox_table = np.array(bbox_table)
np.save('bbox_labels_table.npy', bbox_table)


preds_nms, preds_cls_nms = utils.nms(bbox_cls_preds, bbox_preds, 0.1, 0)
preds_nms_filtered = []
for i in range(len(preds_nms)):
    room = []
    for j in range(len(preds_nms[i])):
        if preds_cls_nms[i][j][0] > 0.3:
            room.append(preds_nms[i][j])
    preds_nms_filtered.append(room)

preds_nms_filtered = np.array(preds_nms_filtered)

np.save('bbox_preds_table_nms.npy', preds_nms_filtered)

