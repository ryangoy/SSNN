import numpy as np
import sys

# /checkpoints/weights.17-2.62.hdf5 is best checkpoint

def combine_2d_3d(cls_2d, loc_2d, cls_3d, loc_3d, K, RT):
	R = RT[:3]
    T = RT[3:].T
    T = np.repeat(T, 2)

    un_t = loc_3d - np.tile(T, (loc.shape[0], 1))
    un_rt_ll = np.dot(un_t[:, :3], R)
    un_rt_ur = np.dot(un_t[:, 3:], R)

    ll_2d = np.dot(un_rt_ll, K.T)
    ur_2d = np.dot(un_rt_ur, K.T)

    ll_2d /= ll_2d[:,-1]
    ur_2d /= ur_2d[:,-1]

    proj_bboxes = np.concatenate([ll_2d, ur_2d], axis=-1)







if __name__ == '__main__':
	if len(sys.argv) < 7:
		print("Usage:\n\tpython combine_2d_3d.py <cls_2d> <loc_2d> <cls_3d> <loc_3d> <K> <RT>")

	cls_2d = np.load(sys.argv[1])
	loc_2d = np.load(sys.argv[2])

	cls_3d = np.load(sys.argv[3])
	loc_3d = np.load(sys.argv[4])

	K = np.load(sys.argv[5])
	RT = np.load(sys.argv[6])

	output = combine_2d_3d(cls_2d, loc_2d, cls_3d, loc_3d, K, RT)