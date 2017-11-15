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
  
  def __init__(self, dims, num_kernels=1, probes_per_kernel=1, probe_steps=10, num_scales=3, ckpt_load=None, ckpt_save=None):
    self.hook_num = 1
    self.dims = dims
    self.probe_steps = probe_steps
    self.probe_size = dims / probe_steps
    self.probe_output = None
    self.ckpt_save = ckpt_save
    self.ckpt_load = ckpt_load

    # Defines self.probe_op
    self.init_probe_op(dims, probe_steps, num_kernels=num_kernels, 
                       probes_per_kernel=probes_per_kernel)

    # Defines self.X_ph, self.y_ph, self.model, self.cost, self.optimizer
    self.init_model(num_kernels, probes_per_kernel, probe_steps, num_scales)

    # Initialize variables
    self.saver = tf.train.Saver()
    self.sess = tf.Session()
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)
    if self.ckpt_load and self.load_checkpoint(self.ckpt_load):
      print('Loaded model from checkpoint successfully.')
    else:
      print('Initialized new model.')

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

  def hook_layer(self, input_layer, reuse=False, activation=None):
    # As defined in Singleshot Multibox Detector, hook layers process
    # intermediates at different scales.

    # Note that we have a linear activation (no activation fn). tf.nn.softmax
    # will be applied to the output.
    if reuse == True:
      scope_name = 'hook' 
    else:
      scope_name = 'hook_' + str(self.hook_num)
    with tf.variable_scope(scope_name) as scope:
      
      # If reuse is True, then we share the weights of the hook layer.
      if reuse and self.hook_num != 1:
        scope.reuse_variables()

      # Predicts the confidence of whether or not an objects exists per feature.
      conf = tf.layers.conv3d(input_layer, filters=2, kernel_size=3, padding='SAME',
                              strides=1, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())

      # Predicts the center coordinate and relative scale of the box
      loc = tf.layers.conv3d(input_layer, filters=6, kernel_size=3, padding='SAME',
                              strides=1, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())

      self.hook_num += 1

      return conf, loc

  def init_model(self, num_kernels, probes_per_kernel, probe_steps, num_scales,
                 learning_rate=0.001, loc_loss_lambda=0.5, reuse_hook=False):

    # Shape: (batches, x_steps, y_steps, z_steps, num_kernels, 
    #         probes_per_kernel)
    self.X_ph = tf.placeholder(tf.float32, (None, probe_steps, probe_steps, 
                                            probe_steps, num_kernels, 
                                            probes_per_kernel,))

    num_features = 0
    dim_size = probe_steps
    for i in range(3):
        num_features += dim_size**3
        dim_size /= 2

    # Shape: (batches, x_steps, y_steps, z_steps)
    self.y_ph_cls = tf.placeholder(tf.int32, (None, num_features, 2))

    self.y_ph_loc = tf.placeholder(tf.float32, (None, num_features, 6))

    # Shape: (batches, x, y, z, features)
    self.dot_product, self.dp_weights = dot_product(self.X_ph, filters=1)

    self.conv1_1 = tf.layers.conv3d(self.dot_product, filters=32, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    self.conv1_2 = tf.layers.conv3d(self.conv1_1, filters=32, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    # First hook layer.
    cls_hook1, loc_hook1 = self.hook_layer(self.conv1_2, reuse=reuse_hook)

    self.pool1 = tf.nn.max_pool3d(self.conv1_2, ksize=[1, 2, 2, 2, 1], 
                                  strides=[1, 2, 2, 2, 1], padding='SAME')

    self.conv2_1 = tf.layers.conv3d(self.pool1, filters=32, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    self.conv2_2 = tf.layers.conv3d(self.conv2_1 , filters=32, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Second hook layer, resolution is 1/2 the first
    cls_hook2, loc_hook2 = self.hook_layer(self.conv2_2, reuse=reuse_hook)

    self.pool2 = tf.nn.max_pool3d(self.conv2_2, ksize=[1, 2, 2, 2, 1], 
                                  strides=[1, 2, 2, 2, 1], padding='SAME')

    self.conv3_1 = tf.layers.conv3d(self.pool2, filters=32, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    self.conv3_2 = tf.layers.conv3d(self.conv3_1 , filters=32, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Third hook layer, resolution is 1/4th the first
    cls_hook3, loc_hook3 = self.hook_layer(self.conv3_2, reuse=reuse_hook)

    self.cls_hooks = [cls_hook1, cls_hook2, cls_hook3]
    self.loc_hooks = [loc_hook1, loc_hook2, loc_hook3]


    cls_hooks_flat = tf.concat([tf.reshape(cls_hook1, (-1, self.conv1_2.shape[1]*self.conv1_2.shape[2]*self.conv1_2.shape[3], 2)),
                               tf.reshape(cls_hook2, (-1, self.conv2_2.shape[1]*self.conv2_2.shape[2]*self.conv2_2.shape[3], 2)),
                               tf.reshape(cls_hook3, (-1, self.conv3_2.shape[1]*self.conv3_2.shape[2]*self.conv3_2.shape[3], 2))],
                               axis=1)
    loc_hooks_flat = tf.concat([tf.reshape(loc_hook1, (-1, self.conv1_2.shape[1]*self.conv1_2.shape[2]*self.conv1_2.shape[3], 6)),
                               tf.reshape(loc_hook2, (-1, self.conv2_2.shape[1]*self.conv2_2.shape[2]*self.conv2_2.shape[3], 6)),
                               tf.reshape(loc_hook3, (-1, self.conv3_2.shape[1]*self.conv3_2.shape[2]*self.conv3_2.shape[3], 6))],
                               axis=1)

    # Define cls loss.
    cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_ph_cls, logits=cls_hooks_flat)
    cls_loss = tf.reduce_mean(cls_loss)

    # Define loc loss.
    diff = self.y_ph_loc - loc_hooks_flat
    # loc_loss_L2 = 0.5*(diff**2)
    # loc_loss_L1 = tf.abs(diff) - 0.5
    # smooth_cond = tf.less(tf.abs(diff), 1.0)
    # loc_loss = tf.where(smooth_cond, loc_loss_L1, loc_loss_L2)
    loc_loss = tf.abs(diff)
    loc_loss = tf.reduce_mean(loc_loss)

    # Combine losses linearly.
    self.loss = cls_loss + loc_loss_lambda * loc_loss

    # Define optimizer.
    self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

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

  def train_val(self, X_trn=None, y_trn_cls=None, y_trn_loc=None, X_val=None, y_val=None, epochs=10, 
                batch_size=4, display_step=10, save_interval=10):
    if X_trn is None:
      X_trn = self.probe_ouput
    assert y_trn_cls is not None and y_trn_loc is not None, "Labels must be defined for train_val call."

    for epoch in range(epochs):
      indices = range(X_trn.shape[0])
      shuffle(indices)
      X_trn = X_trn[indices]
      y_trn_cls = y_trn_cls[indices]
      y_trn_loc = y_trn_loc[indices]

      for step in range(0, X_trn.shape[0], batch_size): 
        batch_x = X_trn[step:step+batch_size]
        batch_y_cls = y_trn_cls[step:step+batch_size]
        batch_y_loc = y_trn_loc[step:step+batch_size]
        _, loss = self.sess.run([self.optimizer, self.loss], feed_dict={self.X_ph: batch_x, 
                                            self.y_ph_cls: batch_y_cls, self.y_ph_loc: batch_y_loc})

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
      if epoch % save_interval and epoch != 0 and self.ckpt_save is not None:
        self.save_checkpoint(self.ckpt_save, epoch)

  def test(self, X_test, save_dir=None, batch_size=1):
    cls_preds = []
    loc_preds = []
    for i in range(0, X_test.shape[0], batch_size):
      batch_x = X_test[i:i+batch_size]
      hooks = self.sess.run(self.cls_hooks + self.loc_hooks, feed_dict={self.X_ph: batch_x})
      cls_preds.append(hooks[:3])
      loc_preds.append(hooks[3:])
    return cls_preds, loc_preds

  def save_checkpoint(self, checkpoint_dir, step, name='ssnn.model'):
    if not isdir(checkpoint_dir):
      makedirs(checkpoint_dir)
    self.saver.save(self.sess, os.path.join(checkpoint_dir, name), global_step=step)

  def load_checkpoint(self, checkpoint_dir):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      ckpt_name = basename(ckpt.model_checkpoint_path)
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
