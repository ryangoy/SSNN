

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt


def plot_bounding_box(bbox, ax, color):

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
    Z = 10.0*Z

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
    collection = Poly3DCollection(verts, facecolors=None, linewidths=1, edgecolors=color, alpha=0.0)
    face_color = color # alternative: matplotlib.colors.rgb2hex([0.5, 0.5, 1])
    collection.set_facecolor(face_color)
    ax.add_collection3d(collection)



def main():
    preds = np.load('category_preds_nms.npy')
    labels = np.load('category_labels.npy')


    for scene_preds, scene_labels in zip(preds, labels):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')


        for pred in scene_preds:
            plot_bounding_box(pred, ax, color='b')

        for label in scene_labels:

            plot_bounding_box(label, ax, color='r')

        plt.show()

if __name__ == '__main__':
    main()