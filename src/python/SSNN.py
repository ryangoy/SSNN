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

  def init_model(self, num_kernels, probes_per_kernel, probe_steps, 
                 learning_rate=0.01):

    # Shape: (batches, num_kernels, probes_per_kernel, x, y, z)
    self.X_ph = tf.placeholder(tf.float32, (None, num_kernels, 
                                            probes_per_kernel, probe_steps, 
                                            probe_steps, probe_steps))

    # Shape: (batches, num_classes), in this case, it's a binary classifer.
    self.y_ph = tf.placeholder(tf.float32, (None, probe_steps, 
                                            probe_steps, probe_steps))
    # self.y_ph = tf.expand_dims(self.y_ph, axis=-1)

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


    # tf.metrics.mean_iou(vox_label, preds, 1)
    self.loss = tf.reduce_mean(tf.square(tf.subtract(self.model, self.y_ph)))
    self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

  def train_val(self, X_trn=None, y_trn=None, X_val=None, y_val=None, epochs=10, 
                batch_size=1, display_step=10):
    if X_trn is None:
      X_trn = self.probe_ouput
    assert y_trn is not None, "Labels must be defined for train_val call."

    for epoch in range(epochs):
      for step in range(0, X_trn.shape[0], batch_size): 
        # batch_x, batch_y = self.get_next_batch(X_trn, y_trn, batch_size)
        batch_x = X_trn[step:step+batch_size]
        batch_y = y_trn[step:step+batch_size]
        self.sess.run(self.optimizer, feed_dict={self.X_ph: batch_x, 
                                            self.y_ph: batch_y})

        if step % display_step == 0:
          loss = self.sess.run(self.loss, 
                             feed_dict={self.X_ph: batch_x, self.y_ph: batch_y})
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
    preds = np.array(preds).squeeze(axis=1)
    if save_dir is not None:
      save_batch(save_dir, batch)
    return preds

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




    



        
      







