import numpy as np
from scipy.misc import imread, imsave
from os import listdir
from os.path import join, isdir, exists
import json

def rgbd2pc_single(rgb_img, d_img, K, RT, index_matrix, KiX=None):
	# KiX = np.linalg.inv(K).dot(index_matrix.T)

	# norms = np.linalg.norm(KiX, ord=2, axis=0)
	# print(KiX.shape)
	# print(norms.shape)
	# KiX /= norms


	d_img = np.right_shift(d_img, 3)
	d_img = d_img.astype(float).flatten()/1000
	rgb_img = rgb_img.reshape((-1, 3))

	cx = K[0,2]
	cy = K[1,2]
	fx = K[0,0]
	fy = K[1,1]

	x = (index_matrix[...,0]-cx) * d_img / fx
	y = (index_matrix[...,1]-cy) * d_img / fy
	z = d_img

	x = np.expand_dims(x, -1)
	y = np.expand_dims(y, -1)
	z = np.expand_dims(z, -1)
	# KiX = np.concatenate([z, y, x], axis=-1)

	KiX = np.concatenate([x, z, -y], axis=-1)



	R = RT[:, :3]
	T = RT[:, 3:]
	Y = KiX
	Y = Y - np.tile(T, (1, Y.shape[0])).T
	Y = Y.dot(np.linalg.inv(R))



	# pc = Y*np.tile(d_img, (1, 3))

	colored_pc = np.concatenate([Y, rgb_img], axis=-1)
	np.savetxt('cpc.txt', colored_pc)


def rgbd2pc(data_path):
	index_matrix = create_index_matrix(530, 730)


	imgs = listdir(data_path)
	imgs = sorted(imgs)
	for img in imgs[6:]:
		folder_path = join(data_path, img)
		if isdir(folder_path):
			
			# extrinsics
			extrinsics_folder = join(folder_path, 'extrinsics')
			if len(listdir(extrinsics_folder)) > 1:
				print('More than one file in extrinsics folder at {}.'.format(extrinsics_folder))
			extrinsics_file = join(extrinsics_folder, listdir(extrinsics_folder)[0])
			extrinsics_npy = np.loadtxt(extrinsics_file)
			anno_extrinsics = extrinsics_npy[:3, :3]

			# fullres_folder = join(folder_path, 'fullres')
			# if not exists(fullres_folder):
			# 	continue
			# rgb_img = None
			# d_img = None
			# intrinsics_npy = None
			# for f in listdir(fullres_folder):
			# 	if f.endswith('.jpg'):
			# 		rgb_img = imread(join(fullres_folder, f))
			# 	elif f.endswith('.png'):
			# 		d_img = imread(join(fullres_folder, f))
			# 	elif f.endswith('.txt'):
			# 		intrinsics_npy = np.loadtxt(join(fullres_folder, f))

			intrinsics_npy = np.loadtxt(join(folder_path, 'intrinsics.txt'))
			for f in listdir(join(folder_path, 'image')):
				rgb_img = imread(join(folder_path, 'image', f))

			for f in listdir(join(folder_path, 'depth_bfx')):
				d_img = imread(join(folder_path, 'depth_bfx', f))

			if rgb_img is None or d_img is None or intrinsics_npy is None:
				print('Image didn\'t load in {}.'.format(folder_path))
				continue

			raw_annotations = json.load(open(join(folder_path, 'annotation3Dfinal', 'index.json')))['objects']


			colored_pc = rgbd2pc_single(rgb_img, d_img, intrinsics_npy, extrinsics_npy, index_matrix)

			bbox_pcs = []


			for raw_annot in raw_annotations:
				if raw_annot is None:
					continue
				for poly in raw_annot['polygon']:
					bbox = annotation_to_bbox(poly, anno_extrinsics)
					bbox_pcs.append(bbox_to_pc(bbox))
				# except:
				# 	continue

			all_bbox_pcs = np.concatenate(bbox_pcs, axis=0)
			bbox_color = np.zeros((all_bbox_pcs.shape))
			bbox_color[:, 0] = 102
			bbox_color[:, 1] = 255
			bbox_color[:, 2] = 102
			all_bbox_pcs = np.concatenate([all_bbox_pcs, bbox_color], axis=-1)

			np.savetxt('bboxes.txt', all_bbox_pcs)
			exit()



def annotation_to_bbox(annotation, R):
	Xs = annotation['X']
	Zs = annotation['Z']
	Ymin = annotation['Ymin']
	Ymax = annotation['Ymax']

	xc = np.concatenate([Xs, Xs], axis=0)
	yc = np.array([Ymin]*len(Xs)+[Ymax]*len(Xs))
	zc = np.concatenate([Zs, Zs], axis=0)

	coords = np.stack([xc, yc, zc])


	# tcoords = np.linalg.inv(R).dot(coords)
	tcoords = coords

	tcoords = np.array([tcoords[0], tcoords[2], -tcoords[1]])
	#tcoords = np.array([tcoords[2], tcoords[0], -tcoords[1]])
	bbox = np.concatenate([np.min(tcoords, axis=1), np.max(tcoords, axis=1)])

	return bbox

def bbox_to_pc(bbox):
	Xs = [0, 3]
	Ys = [1, 4]
	Zs = [2, 5]
	corners = []
	for X in Xs:
		for Y in Ys:
			for Z in Zs:
				corners.append([bbox[X], bbox[Y], bbox[Z]])

	points = []

	for c1 in corners:
		for c2 in corners:
			if c1 is c2:
				continue
			interval = np.tile(np.reshape(np.linspace(0, 1, num=300), (-1,1)), (1, 3))

			edge = (1-interval) * c1 + interval * c2

			points.append(edge)

	points = np.concatenate(points, axis=0)

	return points





def create_index_matrix(width, height):

	# mat = np.zeros((width, height, 2))
	# for i in range(width):
	# 	for j in range(height):
	# 		 mat[i, j, 0] = i
	# 		 mat[i, j, 1] = j
	# flattened = np.array(mat).reshape((width*height, 2))

	# homogenous_coords = np.concatenate([flattened, np.ones((width*height, 1))], axis=-1)
	# return homogenous_coords

	return np.stack(np.meshgrid(np.arange(1, height+1), np.arange(1, width+1)), axis=-1).reshape((-1, 2))


if __name__ == '__main__':
	rgbd2pc('/home/ryan/cs/datasets/SUNRGBD/kv2/align_kv2/')
