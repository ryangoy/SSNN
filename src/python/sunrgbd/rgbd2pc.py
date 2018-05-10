import numpy as np
from scipy.misc import imread, imsave
from os import listdir
from os.path import join, isdir, exists

def rgbd2pc_single(rgb_img, d_img, K, RT, index_matrix, KiX=None):

	KiX = np.linalg.inv(K).dot(index_matrix.T)

	R = RT[:, :3]
	T = RT[:, 3:]

	Y = KiX - np.tile(T, (1, KiX.shape[1]))
	Y = R.dot(KiX).T

	d_img = d_img.reshape((-1, 1))
	rgb_img = rgb_img.reshape((-1, 3))

	pc = Y*np.tile(d_img, (1, 3))

	colored_pc = np.concatenate([pc, rgb_img], axis=-1)
	print(colored_pc.shape)
	np.savetxt('cpc.txt', colored_pc[::3])
	print("saved")
	exit()



def rgbd2pc(data_path):
	index_matrix = create_index_matrix(1280, 1920)

	for img in listdir(data_path)[59:]:
		folder_path = join(data_path, img)
		if isdir(folder_path):
			
			# extrinsics
			extrinsics_folder = join(folder_path, 'extrinsics')
			if len(listdir(extrinsics_folder)) > 1:
				print('More than one file in extrinsics folder at {}.'.format(extrinsics_folder))
			extrinsics_file = join(extrinsics_folder, listdir(extrinsics_folder)[0])
			extrinsics_npy = np.loadtxt(extrinsics_file)

			fullres_folder = join(folder_path, 'fullres')
			if not exists(fullres_folder):
				continue
			rgb_img = None
			d_img = None
			intrinsics_npy = None
			for f in listdir(fullres_folder):
				if f.endswith('.jpg'):
					rgb_img = imread(join(fullres_folder, f))
				elif f.endswith('.png'):
					d_img = imread(join(fullres_folder, f))
				elif f.endswith('.txt'):
					intrinsics_npy = np.loadtxt(join(fullres_folder, f))
			if rgb_img is None or d_img is None or intrinsics_npy is None:
				print('Image didn\'t load in {}.'.format(fullres_folder))

			colored_pc = rgbd2pc_single(rgb_img, d_img, intrinsics_npy, extrinsics_npy, index_matrix)


def create_index_matrix(width, height):

	mat = np.zeros((width, height, 2))
	for i in range(width):
		for j in range(height):
			 mat[i, j, 0] = i
			 mat[i, j, 1] = j
	flattened = np.array(mat).reshape((width*height, 2))

	homogenous_coords = np.concatenate([flattened, np.ones((width*height, 1))], axis=-1)
	return homogenous_coords


if __name__ == '__main__':
	rgbd2pc('/home/ryan/cs/datasets/SUNRGBD/kv2/align_kv2/')
