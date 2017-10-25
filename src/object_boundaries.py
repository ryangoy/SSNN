## Stores boundaries of objects and their classes in parallel array
## Objects are stored as [center_x, center_y, center_z, width, height, depth]
## Also keeps track of metadata and lists any txt files that caused errors during runtime

import os
import numpy as np


def generate_bounding_boxes(pointcloud):
    """
    Generates bounding box labels from semantic labels.

    Args:
      pointcloud (np.array): input semantic labels with shape (batches, sample, points, xyzrgb).
            Since number of samples per scene are variable and number of points are variable, 
            it is a numpy array of numpy arrays of numpy ndarrays, i.e. to access an object, use
            pointcloud[scene_id][object_id].

    Returns:
      bounding_boxes (np.array): output bounding box labels with shape (batches, sample, min_x/min_y/min_z/max_x/max_y/max_z).
            Since samples is not constant per scene, output in a fashion similar to the input.
    """

    return np.array([np.array([np.array(create_bounds(pointcloud[i][j])) for j in range(len(pointcloud[i]))]) for i in range(len(pointcloud))])

def create_bounds(data):
    x_vals = data[:, 0]
    max_x = max(x_vals)
    min_x = min(x_vals)
    y_vals = data[:, 1]
    max_y = max(y_vals)
    min_y = min(y_vals)
    z_vals = data[:, 2]
    max_z = max(z_vals)
    min_z = min(z_vals)

    return [min_x, min_y, min_z, max_x, max_y, max_z]

# create a label for a given point cloud representation of an object
def create_label(data):
    # find bounding points in each dimension
    min_x, min_y, min_z, max_x, max_y, max_z = create_bounds(data)

    # store midpoint of two ends, along with the length along each dimension
    bbox = [(max_x + min_x) / 2.0, (max_y + min_y) / 2.0, (max_z + min_z) / 2.0, max_x - min_x, max_y - min_y, max_z - min_z]
    return bbox


if __name__ =='__main__':
    boxes = []
    labels = []
    metadata = []
    error_files = []

    # iterate over the files
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
                        data = np.genfromtxt(dirName + '/Annotations/' + fname).astype('float')
                    except ValueError:
                        print('error on', area, room, obj_numbered)
                        error_files.append([area, room, fname])
                        continue
                
                    bbox = create_label(data)	
                    boxes.append(bbox)  # store bbox
                    labels.append(obj)  # store label
                    metadata.append([area, room, obj_numbered])	 # store area, room, and name of file 		

    # save our arrays
    np.save('../../../bboxes/boxes.npy', np.array(boxes))
    np.save('../../../bboxes/labels.npy', np.array(labels))
    np.save('../../../bboxes/metadata.npy', np.array(metadata))
    print('errors:', error_files)
