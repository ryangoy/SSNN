

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
    for i in range(8): Z[i,:] = points[i,:]
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
    preds = np.load('category_preds_nms.npy')
    labels = np.load('category_labels.npy')

    pred_vols = []
    label_vols = []
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

    plt.hist(pred_vols, bins=25, color='b', alpha=0.5, range=(0, 100))
    plt.hist(label_vols, bins=25, color='r', alpha=0.5, range=(0,100))
    plt.show()
    
    plt.hist(np.array(pred_vols)/np.array(label_vols), range=(0, 5), bins=20, color='b')
    plt.show()


if __name__ == '__main__':
    plot_3d_bboxes()
