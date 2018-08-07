import numpy as np
import sys
import scipy.io as sio

def generate_submission(preds, confs, save_path):
	preds = np.load(preds)
	confs = np.load(confs)

	submission = []
	for scene_preds, scene_confs in zip(preds, confs):
		scene_info = []
		for sp, sc in zip(scene_preds, scene_confs):
			centroid = sp[:3]
			coeffs = sp[3:6]
			th = sp[6]
			basis = np.array([[np.cos(th), -np.sin(th), 0],
				              [np.sin(th), np.cos(th), 0],
				              [0, 0, 1]])
			scene_info.append([[centroid, basis, coeffs, np.array(sc)]])
		submission.append(np.array(scene_info))
	sub_dict = {'boxes': submission}
	sio.savemat(save_path, sub_dict)

if __name__ == '__main__':
	if len(sys.argv) < 4:
		print('Usage:\n\n\tpython {} <bbox predictions> <bbox confidences> <save path>'.format(sys.argv[0]))

	generate_submission(sys.argv[1], sys.argv[2], sys.argv[3])