# visualization of prediction boxes (blue) and label boxes (red) in normalized coordinate space

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt

def plot_bounding_box(bbox, ax, color):
    volume = (bbox[3]-bbox[0])*(bbox[4]-bbox[1])*(bbox[5]-bbox[2])
    volume = abs(volume)
    
    points = np.array([[bbox[0], bbox[1], bbox[2]],
                       [bbox[0], bbox[4], bbox[2]],
                       [bbox[0], bbox[4], bbox[5]],
                       [bbox[0], bbox[1], bbox[5]],
                       [bbox[3], bbox[1], bbox[2]],
                       [bbox[3], bbox[4], bbox[2]],
                       [bbox[3], bbox[4], bbox[5]],
                       [bbox[3], bbox[1], bbox[5]]])

    Z = np.zeros((8,3))

    for i in range(8): 
        Z[i,:] = points[i,:]


    ax.scatter3D(Z[:, 0], Z[:, 1], Z[:, 2])

    # list of sides' polygons of figure
    verts = [[Z[0],Z[1],Z[2],Z[3]],
     [Z[4],Z[5],Z[6],Z[7]], 
     [Z[0],Z[1],Z[5],Z[4]], 
     [Z[2],Z[3],Z[7],Z[6]], 
     [Z[1],Z[2],Z[6],Z[5]],
     [Z[4],Z[7],Z[3],Z[0]], 
     [Z[2],Z[3],Z[7],Z[6]]]

    # plot sides
    collection = Poly3DCollection(verts, facecolors=None, linewidths=1, edgecolors=color, alpha=0.1)
    face_color = color # alternative: matplotlib.colors.rgb2hex([0.5, 0.5, 1])
    collection.set_facecolor(face_color)
    ax.add_collection3d(collection)
    return volume

def plot_3d_bboxes():
    preds = np.load('../category_preds_nms.npy')
    labels = np.load('../category_labels.npy')

    pred_vols = []
    label_vols = []
    scene_id = 0
    for scene_preds, scene_labels in zip(preds, labels):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_xlim(0,8)
        ax.set_ylim(0, 8)
        ax.set_zlim(0, 8)

        #for pred in scene_preds:
        print('Scene {}'.format(scene_id))
        find_IoU(scene_preds, scene_labels)
        for i in range(len(scene_labels)):
#            if len(scene_labels) > 1:
#                break
            v = plot_bounding_box(scene_preds[i], ax, color='b')
            pred_vols.append(v)
#            break

        for label in scene_labels:
#            if len(scene_labels) > 1:
#                break
            v = plot_bounding_box(label, ax, color='r')
            label_vols.append(v)
            
        plt.show()
        scene_id+=1 
    plt.hist(pred_vols, bins=25, color='b', alpha=0.5, range=(0, 100))
    plt.hist(label_vols, bins=25, color='r', alpha=0.5, range=(0,100))
    plt.show()
    
    plt.hist(np.array(pred_vols)/np.array(label_vols), range=(0, 5), bins=20, color='b')
    plt.show()


def find_IoU(preds, labels):
    print(len(preds))
    print(len(labels))
    matched_labels=[]
    for p in range(len(labels)):
        pred = preds[p]
        # Find a label that the prediction corresponds to if it exists.
        pred_matched = False  
        
        for l in range(len(labels)):

            label = labels[l]
            if l in matched_labels:
                continue

            max_LL = np.max(np.array([pred[:3], label[:3]]), axis=0)
            
            min_UR = np.min(np.array([pred[3:], label[3:]]), axis=0)
            print('pred_coords: {}'.format(pred))
            print('label_coords: {}'.format(label))
            print('MaxLL: {}'.format(max_LL))
            print('MinUr: {}'.format(min_UR))
            intersection = max(0, np.prod(min_UR - max_LL))
            print('intersection: {}'.format(intersection))

            union = np.prod(pred[3:]-pred[:3]) + np.prod(label[3:]-label[:3]) - intersection

            print('union: {}'.format(union))

            # If we found a label that matches our prediction, i.e. a true positive
            if min(min_UR - max_LL) > 0:
                print(intersection/union)
                pred_matched = True
                matched_labels.append(l)
                break


if __name__ == '__main__':
    plot_3d_bboxes()
