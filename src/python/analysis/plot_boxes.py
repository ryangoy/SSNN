# visualization of prediction boxes (blue) and label boxes (red) in normalized coordinate space

import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import matplotlib.pyplot as plt

def plot_bounding_box(bbox, ax, color):

    # define the coordinates of the corners (order matters)
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
    collection = Poly3DCollection(verts, facecolors=None, linewidths=1, edgecolors=color, alpha=0.0)
    face_color = color # alternative: matplotlib.colors.rgb2hex([0.5, 0.5, 1])
    collection.set_facecolor(face_color)
    ax.add_collection3d(collection)


def plot_3d_bboxes():
#    preds = np.load('bbox_preds_table_nms.npy')
#    labels = np.load('bbox_labels_table.npy')
    rooms = np.load('test_rooms.npy')
    skip_factor = 20 # only plot 1 in 20 points in a room (saves time)

#    for scene, scene_preds, scene_labels in zip(rooms, preds, labels):
#        fig = plt.figure()
#        ax = fig.add_subplot(111, projection='3d')

#        ax.set_xlabel('X')
#        ax.set_ylabel('Y')
#        ax.set_zlabel('Z')

#        ax.set_xlim(0, 8)
#        ax.set_ylim(0, 8)
#        ax.set_zlim(0, 8)

        # plot as many predictions as we have
#        for i in range(len(scene_preds)):
#            plot_bounding_box(scene_preds[i], ax, color='b')
#        for label in scene_labels:
#            plot_bounding_box(label, ax, color='r')

#        ax.scatter(scene[::skip_factor, 0], scene[::skip_factor, 1], scene[::skip_factor, 2], s=1, color='g')
#        plt.show()

if __name__ == '__main__':
    plot_3d_bboxes()
