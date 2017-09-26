## Stores boundaries of objects and their classes in parallel array
## Objects are stored as [center_x, center_y, center_z, width, height, depth]
## Also keeps track of metadata and lists any txt files that caused errors during runtime

import os
import numpy as np

boxes = []
labels = []
metadata = []
error_files = []


for dirName, subdirList, fileList in os.walk('../../../Stanford3dDataset_v1.2'):
	if 'Annotations' in subdirList:
		path = dirName.split('/')
		area = path[4]
		room = path[5]
		print('new room', room)
		object_fnames = os.listdir(dirName + '/Annotations/')
		for fname in object_fnames:
			if fname.endswith('.txt'):
				print(fname)
				obj = fname.split('_')[0]
				obj_numbered = fname.split('.')[0]
				
				try:
					data = np.loadtxt(dirName + '/Annotations/' + fname)
				except ValueError:
					try:
						data = np.genfromtxt(dirName + '/Annotations/' + fname).astype('float')
					except ValueError:
						print('error on', area, room, obj_numbered)
						error_files.append([area, room, fname])
						continue

				x_vals = data[:, 0]
				max_x = max(x_vals)
				min_x = min(x_vals)
				y_vals = data[:, 1]
				max_y = max(y_vals)
				min_y = min(y_vals)
				z_vals = data[:, 2]
				max_z = max(z_vals)
				min_z = min(z_vals)
				
				bbox = [(max_x + min_x) / 2.0, (max_y + min_y) / 2.0, (max_z + min_z) / 2.0, max_x - min_x, max_y - min_y, max_z - min_z]
				boxes.append(bbox)
				labels.append(obj)
				metadata.append([area, room, obj_numbered])			

np.save('../../../bboxes/boxes.npy', np.array(boxes))
np.save('../../../bboxes/labels.npy', np.array(labels))
np.save('../../../bboxes/metadata.npy', np.array(metadata))
print('errors:', error_files)
