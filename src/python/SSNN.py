import numpy as np
import tensorflow as tf
from tf_ops import *

class SSNN:
  
  def __init__(self, dims, num_kernels=1, probes_per_kernel=1, probe_steps=10):

    # Defines self.probe_op
    self.init_probe_op(dims, probe_steps, num_kernels=num_kernels, 
                       probes_per_kernel=probes_per_kernel)

    # Defines self.X_ph, self.y_ph, self.model, self.cost, self.optimizer
    self.init_model(num_kernels, probes_per_kernel, probe_steps)

    self.sess = tf.Session()
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

    self.dims = dims
    self.probe_steps = probe_steps
    self.probe_size = dims / probe_steps


    self.probe_output = None


  def fc_layer(self, x, num_nodes, name='fc_layer', activation=tf.nn.relu):
    with tf.variable_scope(name, reuse=False):
      W = tf.Variable(shape=[x.shape[-1], num_nodes], 
                      initializer=tf.contrib.layers.xavier_initializer())
      b = tf.Variable(tf.zeros(shape=[num_nodes]))
      output = tf.nn.bias_add(tf.matmul(x, W), b)
      if activation is not None:
        output = activation(output)
      return output

  def dropout(self, x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

  def init_probe_op(self, dims, steps, num_kernels=1, probes_per_kernel=1):
    """
    The idea behind having a separate probe op is that we are converting from
    continuous space to discrete space here. Running backprop on this layer and
    having it every iteration will be too slow, so hopefully the dot product
    layer will be sufficient. Otherwise, we can look more into optimizing the
    probe layer.

    Args:
      input_dims: tuple [x_meters, y_meters, z_meters]
      step_size: tuple [x_stride, y_stride, z_stride]
    """
    if type(steps) is int:
      steps = [steps, steps, steps]
    else:
      assert len(steps) == 3, \
          "Must have a step size for each xyz dimension, or input an int."

    # Shape: (batches, num_points, xyz)
    self.points_ph = tf.placeholder(tf.float32, (None, None, 3))

    # Shape: (batches, probes, samples per probe, x, y, z)
    self.probe_op = probe3d(self.points_ph, dims, 
                            steps=steps, 
                            num_kernels=num_kernels, 
                            probes_per_kernel=probes_per_kernel)

  def init_model(self, num_kernels, probes_per_kernel, probe_steps, learning_rate=0.01):

    # Shape: (batches, num_kernels, probes_per_kernel, x, y, z)
    self.X_ph = tf.placeholder(tf.float32, 
        (None, num_kernels, probes_per_kernel, probe_steps, probe_steps, probe_steps))

    # Shape: (batches, num_classes), in this case, it's a binary classifer.
    self.y_ph = tf.placeholder(tf.float32, (None, None, 4))

    # Shape: (batches, probes, x, y, z)
    self.dot_product = dot_product(self.X_ph)

    self.dot_product = tf.transpose(self.dot_product, (0, 2, 3, 4, 1))

    self.c1 = tf.layers.conv3d(self.dot_product, filters=16, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    self.mp1 = tf.nn.max_pool3d(self.c1, ksize=[1, 2, 2, 2, 1], 
                                  strides=[1, 1, 1, 1, 1], padding='SAME')

    self.model = tf.layers.conv3d(self.mp1, filters=1, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Find box size
    box_size = dims / steps

    # 
    self.loss = self.IoU_loss()
    self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

  def train_val(self, X_trn=None, y_trn=None, X_val=None, y_val=None, epochs=10, 
                batch_size=1, display_step=10):
    if X_trn is None:
      X_trn = self.probe_ouput
    assert y_trn is not None, "Labels must be defined for train_val call."

    for epoch in range(epochs):
      for step in range(int(X_trn.shape[0]/batch_size)):
        batch_x, batch_y = self.get_next_batch(X_trn, y_trn, batch_size)
        sess.run(self.optimizer, feed_dict={self.X_ph: batch_x, 
                                            self.y_ph: batch_y})

        if step % display_step == 0:
          loss, acc = self.sess.run([self.cost, self.accuracy], 
                                feed_dict={x: batch_x, y: batch_y})
          print("Iter {}, Batch Loss={:.6f}, Training Accuracy={:.5f}.".format(step, loss, acc))

      if X_val is not None and y_val is not None:
        loss = sess.run(self.cost, feed_dict={x: X_val, y: y_val})

      print("Epoch {}, Validation Loss={:6f}, Validation Accuracy={:.5f}.".format(loss, acc))

  def probe(self, X):
    """
    Args:
      X (np.ndarray): array of pointclouds (batches, num_points, 3)
    """
    pcs = []
    for pc in X:
      pc_disc = self.sess.run(self.probe_op, feed_dict={self.points_ph: pc})
      pcs.append(pc_disc)
    self.probe_output = pcs
    return np.array(pcs)

  def IoU_loss(self, preds, labels):
    """
    Args:
      preds (tensor): predicted confidence value for a certain box (batches, x, y, z)
      labels (tensor): labeled boxes with (batches, box, 6), with the format for
                       a box being min_x, min_y, min_z, max_x, max_y, max_z
    """

    vox_label = np.zeros((preds.shape))

    for batch_id in labels.shape[0]:
      for bbox in labels[batch_id]:
        # bbox is [min_x, min_y, max_x, max_y]
        c1 = np.floor(bbox[:3] / self.probe_size)
        c2 = np.ceil(bbox[3:] / self.probe_size)

        diff = c2 - c1

        for i in range(diff[0]):
          for j in range(diff[1]):
            for k in range(diff[2]):
              label_coords = c1 + [i,j,k]
              if vox_label[label_coords] != 0:
              LL = np.max([bbox[:3]/self.probe_size, label_coords], axis=0)
              UR = np.min([bbox[3:]/self.probe_size, label_coords+1], axis=0) 
              intersection = np.sqrt(np.sum(np.square([LL, UR])))
              vox_label[label_coords] = max(intersection, vox_label[label_coords])

    return tf.metrics.mean_iou(vox_label, preds, 1)


    



        
      







