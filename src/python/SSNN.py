import numpy as np
import tensorflow as tf
import _probe_3d_grad
from tf_ops import *

class SSNN:
  
  def __init__(self, input_dims):

    # Defines self.X_ph, self.y_ph, self.model, self.cost, self.optimizer
    self.init_model(input_dims, step_size)
    self.sess = tf.Session()
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

  def max_pool(self, x, kshape, name='conv2d'):
    """
    Args:
      x: an input tensor
      kshape: a tuple with shape 
        [batch_step, width_step, height_step, feature_step]
      name: name of layer
    """
    return tf.nn.max_pool(x, ksize=kshape, strides=kshape, padding='SAME', name=name)

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

  def init_model(self, input_dims, step_size, learning_rate=0.01, num_probes=1, num_classes=2):
    """
    Args:
      input_dims: tuple [x_meters, y_meters, z_meters]
      step_size: tuple [x_stride, y_stride, z_stride]
    """
    if type(step_size) is int:
      step_size = [step_size, step_size, step_size]
    else:
      assert len(step_size) == 3, "Must have a step size for each xyz dimension, or input an int."

    # Shape: (batches, num_points, rgbxyz)
    self.X_ph = tf.placeholder(tf.float32, (None, None, 6))

    # Shape: (batches, num_classes), in this case, it's a binary classifer.
    self.y_ph = tf.placeholder(tf.float32, [None, num_classes])
    
    # Shape: (batches, probes, samples per probe, x, y, z)
    self.model = probe3d(self.X, stride=step_size, num_probes=num_probes)

    # Shape: (batches, probes, x, y, z)
    self.model = dot_product(self.model)

    self.model = tf.transpose(self.model, (0, 2, 3, 4, 1))

    self.model = tf.layers.conv3d(self.model, filters=16, kernel_size=3, strides=1, padding='SAME',
      activation=tf.nn.relu, kernel_initializer=tf.contrib.layers.xavier_initializer())

    self.model = tf.nn.max_pool3d(self.model, ksize=[1, 2, 2, 2, 1], strides=[1, 1, 1, 1, 1], padding='SAME')

    # Repeat more 3d convolutions
    # TO DO

    self.model = tf.flatten(self.model)

    # Linear activation.
    self.model = self.fc_layer(self.model, num_classes)

    # TO DO: Implement IoU loss
    # Probability error for each class, which is assumed to be independent.
    self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
                                labels=self.y, logits=self.model))
    self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

  def train_val(self, X_trn, y_trn, X_val=None, y_val=None, epochs=10, 
                batch_size=1, display_step=100):

    for epoch in range(epochs):
      for step in range(int(X_trn.shape[0]/batch_size)):
        batch_x, batch_y = self.get_next_batch(X_trn, y_trn, batch_size)
        sess.run(self.optimizer, feed_dict={self.X_ph: batch_x, self.y_ph: batch_y})

        if step % display_step == 0:
          loss, acc = self.sess.run([self.cost, self.accuracy], 
                                feed_dict={x: batch_x, y: batch_y})
          print("Iter {}, Batch Loss={:.6f}, Training Accuracy={:.5f}.".format(step, loss, acc))

      if X_val is not None and y_val is not None:
        loss = sess.run(self.cost, feed_dict={x: X_val, y: y_val})

      print("Epoch {}, Validation Loss={:6f}, Validation Accuracy={:.5f}.".format(loss, acc))

