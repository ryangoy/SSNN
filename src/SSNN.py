####################################################
# Defines the SSNN model.
#                        
# @author Ryan Goy
####################################################

import numpy as np
import tensorflow as tf
from tf_ops import *
from random import shuffle

class SSNN:
  
  def __init__(self, dims, num_kernels=1, probes_per_kernel=1, probe_steps=10):

    # Defines self.probe_op
    self.init_probe_op(dims, probe_steps, num_kernels=num_kernels, 
                       probes_per_kernel=probes_per_kernel)

    # Defines self.X_ph, self.y_ph, self.model, self.cost, self.optimizer
    self.init_model(num_kernels, probes_per_kernel, probe_steps)

    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True

    # Initialize variables
    # TODO: add support for checkpoints
    self.sess = tf.Session()
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

    self.dims = dims
    self.probe_steps = probe_steps
    self.probe_size = dims / probe_steps

    self.probe_output = None

  def fc_layer(self, x, num_nodes, name='fc_layer', activation=tf.nn.elu):
    """
    Dense layer.
    """
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

    After self.init_probe_op is called, self.probe can be called to run the op.

    Args:
      input_dims: tuple [x_meters, y_meters, z_meters]
      step_size: int or tuple [x_stride, y_stride, z_stride]
    """
    if type(steps) is int:
      steps = [steps, steps, steps]
    else:
      assert len(steps) == 3, \
          "Must have a step size for each xyz dimension, or input an int."

    # Shape: (batches, num_points, xyz)
    self.points_ph = tf.placeholder(tf.float32, (None, None, 3))

    # Shape: (batches, x, y, z, probes, samples per probe)
    self.probe_op = probe3d(self.points_ph, dims, 
                            steps=steps, 
                            num_kernels=num_kernels, 
                            probes_per_kernel=probes_per_kernel)

  def init_model(self, num_kernels, probes_per_kernel, probe_steps, 
                 learning_rate=0.0001):

    # Shape: (batches, x_steps, y_steps, z_steps, num_kernels, 
    #         probes_per_kernel)
    self.X_ph = tf.placeholder(tf.float32, (None, probe_steps, probe_steps, 
                                            probe_steps, num_kernels, 
                                            probes_per_kernel,))

    # Shape: (batches, x_steps, y_steps, z_steps)
    self.y_ph = tf.placeholder(tf.float32, (None, probe_steps, 
                                            probe_steps, probe_steps))

    # Shape: (batches, x, y, z, features)
    self.dot_product, self.dp_weights = dot_product(self.X_ph, filters=1)

    self.c1 = tf.layers.conv3d(self.dot_product, filters=16, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
    # print self.c1.shape
    # self.mp1 = tf.nn.max_pool3d(self.c1, ksize=[1, 2, 2, 2, 1], 
    #                               strides=[1, 2, 2, 2, 1], padding='SAME')
    # print self.mp1.shape
    self.model = tf.layers.conv3d(self.c1, filters=1, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
    self.model = tf.squeeze(self.model, -1)
    self.loss = tf.reduce_mean(tf.square(tf.subtract(self.model, self.y_ph)))
    self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

  def train_val(self, X_trn=None, y_trn=None, X_val=None, y_val=None, epochs=10, 
                batch_size=4, display_step=10):
    if X_trn is None:
      X_trn = self.probe_ouput
    assert y_trn is not None, "Labels must be defined for train_val call."

    for epoch in range(epochs):
      indices = range(X_trn.shape[0])
      shuffle(indices)
      X_trn = X_trn[indices]
      y_trn = y_trn[indices]

      for step in range(0, X_trn.shape[0], batch_size): 
        batch_x = X_trn[step:step+batch_size]
        batch_y = y_trn[step:step+batch_size]
        _, loss, xph, intermediate = self.sess.run([self.optimizer, self.loss, self.X_ph, self.dot_product], feed_dict={self.X_ph: batch_x, 
                                            self.y_ph: batch_y})

        if step % display_step == 0:
          print("Epoch: {}, Iter: {}, Loss: {:.6f}.".format(epoch, step, loss))

      if X_val is not None and y_val is not None:
        val_loss = 0
        for step in range(0, X_val.shape[0], batch_size):
          val_batch_x = X_val[step:step+batch_size]
          val_batch_y = y_val[step:step+batch_size]
          val_loss += self.sess.run(self.loss, 
                      feed_dict={self.X_ph: val_batch_x, sef.y_ph: val_batch_y})

        print("Epoch: {}, Validation Loss: {:6f}.".format(epoch, 
                                                       val_loss/X_val.shape[0]))
      

      

  def test(self, X_test, save_dir=None, batch_size=1):
    preds = []
    for i in range(0, X_test.shape[0], batch_size):
      batch_x = X_test[i:i+batch_size]
      batch = self.sess.run(self.model, feed_dict={self.X_ph: batch_x})
      preds.append(batch)
    preds = np.array(preds)
    preds = preds.reshape((-1,) + preds.shape[2:])
    return preds

  def probe(self, X):
    """
    Args:
      X (np.ndarray): array of pointclouds (batches, num_points, 3)
    """
    pcs = []
    counter = 0
    for pc in X:
      pc = np.array([pc[0]])
      pc_disc = self.sess.run(self.probe_op, feed_dict={self.points_ph: pc})
      pcs.append(pc_disc)
      if counter % 10 == 0 and counter != 0:
        print('Finished probing {} pointclouds'.format(counter))
      counter += 1
    self.probe_output = pcs
    return np.array(pcs)




    



        
      







