####################################################
# Defines the SSNN model.
#                        
# @author Ryan Goy
####################################################

import numpy as np
import tensorflow as tf
from tf_ops import *
from random import shuffle
from os.path import isdir, join
from os import makedirs
import psutil
import os

class SSNN:
  
  def __init__(self, dims, num_kernels=1, probes_per_kernel=1, dot_layers=8, probe_steps=32, probe_hook_steps=16, 
               num_scales=3, ckpt_load=None, ckpt_save=None, loc_loss_lambda=1, learning_rate=0.001, k_size_factor=3,
               num_classes=2):

    self.hook_num = 1
    self.dims = dims
    self.probe_steps = probe_steps
    self.probe_size = dims / probe_steps
    self.probe_output = None
    self.ckpt_save = ckpt_save
    self.ckpt_load = ckpt_load
    self.num_kernels = num_kernels
    self.probes_per_kernel = probes_per_kernel

    # Defines self.probe_op
    self.init_probe_op(dims, probe_steps, num_kernels=num_kernels, 
                       probes_per_kernel=probes_per_kernel, k_size_factor=k_size_factor)

    # Defines self.X_ph, self.y_ph, self.model, self.cost, self.optimizer
    self.init_model(num_kernels, probes_per_kernel, probe_steps, probe_hook_steps, num_scales, num_classes, dot_layers=dot_layers, loc_loss_lambda=loc_loss_lambda, learning_rate=learning_rate)

    # Initialize variables
    self.saver = tf.train.Saver()
    self.sess = tf.Session()
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

    if self.ckpt_load and self.load_checkpoint(self.ckpt_load):
      print('Loaded SSNN model from checkpoint successfully.')
    else:
      print('Initialized new SSNN model.')

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

  def init_probe_op(self, dims, steps, num_kernels=1, probes_per_kernel=1, k_size_factor=3):
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
                            probes_per_kernel=probes_per_kernel,
                            k_size_factor=k_size_factor)

  def hook_layer(self, input_layer, reuse=False, activation=None, dropout=0.1, num_classes=2):
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

      input_layer = tf.layers.conv3d(input_layer, filters=16, kernel_size=1, padding='SAME',
                              strides=1, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())
      # input_layer = tf.nn.dropout(input_layer, dropout)
      #input_layer = tf.nn.dropout(input_layer, dropout)
      # Predicts the confidence of whether or not an objects exists per feature.
      conf = tf.layers.conv3d(input_layer, filters=num_classes, kernel_size=1, padding='SAME',
                              strides=1, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())

      # Predicts the center coordinate and relative scale of the box
      loc = tf.layers.conv3d(input_layer, filters=6, kernel_size=1, padding='SAME',
                              strides=1, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())

      self.hook_num += 1

      return conf, loc

  def init_model(self, num_kernels, probes_per_kernel, probe_steps, probe_hook_steps, num_scales, num_classes,
                 learning_rate=0.0001, loc_loss_lambda=1, dot_layers=8, reuse_hook=False, dropout=0.05):

    # Shape: (batches, x_steps, y_steps, z_steps, num_kernels, 
    #         probes_per_kernel)
    self.X_ph = tf.placeholder(tf.float32, (None, probe_steps, probe_steps, 
                                            probe_steps, num_kernels, 
                                            probes_per_kernel))

    num_features = 0

    dim_size = probe_steps
    p_dim_size= probe_hook_steps
    num_p_features = 0
    for i in range(3):
        num_features += dim_size**3
        num_p_features += p_dim_size**3
        p_dim_size /= 2
        dim_size /= 2

    # Shape: (batches, x_steps, y_steps, z_steps)
    self.y_ph_cls = tf.placeholder(tf.int32, (None, num_p_features, num_classes))
    self.y_ph_loc = tf.placeholder(tf.float32, (None, num_p_features, 6))

    # Shape: (batches, x, y, z, features)
    self.dot_product, self.dp_weights = dot_product(self.X_ph, filters=dot_layers)


    self.conv0_1 = tf.layers.conv3d(self.dot_product, filters=32, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
    # self.conv0_1 = tf.nn.dropout(self.conv0_1, dropout)
    self.conv0_2 = tf.layers.conv3d(self.conv0_1, filters=32, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())


    self.pool0 = tf.nn.max_pool3d(self.conv0_2, ksize=[1, 2, 2, 2, 1], 
                                  strides=[1, 2, 2, 2, 1], padding='SAME')
    
    self.conv1_1 = tf.layers.conv3d(self.pool0, filters=32, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
    # self.conv0_1 = tf.nn.dropout(self.conv0_1, dropout)
    self.conv1_2 = tf.layers.conv3d(self.conv1_1, filters=32, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())


    self.pool1 = tf.nn.max_pool3d(self.conv1_2, ksize=[1, 2, 2, 2, 1], 
                                  strides=[1, 2, 2, 2, 1], padding='SAME')



    self.conv2_1 = tf.layers.conv3d(self.pool1, filters=32, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
    # self.conv1_1 = tf.nn.dropout(self.conv1_1, dropout)
    self.conv2_2 = tf.layers.conv3d(self.conv2_1, filters=32, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())


    # First hook layer.
    cls_hook1, loc_hook1 = self.hook_layer(self.conv2_2, reuse=reuse_hook, num_classes=num_classes)

    self.pool2 = tf.nn.max_pool3d(self.conv2_2, ksize=[1, 2, 2, 2, 1], 
                                  strides=[1, 2, 2, 2, 1], padding='SAME')

    self.conv3_1 = tf.layers.conv3d(self.pool2, filters=64, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    self.conv3_2 = tf.layers.conv3d(self.conv3_1 , filters=64, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Second hook layer, resolution is 1/2 the first
    cls_hook2, loc_hook2 = self.hook_layer(self.conv3_2, reuse=reuse_hook, num_classes=num_classes)

    self.pool3 = tf.nn.max_pool3d(self.conv3_2, ksize=[1, 2, 2, 2, 1], 
                                  strides=[1, 2, 2, 2, 1], padding='SAME')

    self.conv4_1 = tf.layers.conv3d(self.pool3, filters=64, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    self.conv4_2 = tf.layers.conv3d(self.conv4_1 , filters=64, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Third hook layer, resolution is 1/4th the first
    cls_hook3, loc_hook3 = self.hook_layer(self.conv4_2, reuse=reuse_hook, num_classes=num_classes)

    self.cls_hooks = [cls_hook1, cls_hook2, cls_hook3]
    self.loc_hooks = [loc_hook1, loc_hook2, loc_hook3]

    cls_hooks_flat = tf.concat([tf.reshape(cls_hook1, (-1, self.conv2_2.shape.as_list()[1]*self.conv2_2.shape.as_list()[2]*self.conv2_2.shape.as_list()[3], num_classes)),
                               tf.reshape(cls_hook2, (-1, self.conv3_2.shape.as_list()[1]*self.conv3_2.shape.as_list()[2]*self.conv3_2.shape.as_list()[3], num_classes)),
                               tf.reshape(cls_hook3, (-1, self.conv4_2.shape.as_list()[1]*self.conv4_2.shape.as_list()[2]*self.conv4_2.shape.as_list()[3], num_classes))],
                               axis=1)
    loc_hooks_flat = tf.concat([tf.reshape(loc_hook1, (-1, self.conv2_2.shape.as_list()[1]*self.conv2_2.shape.as_list()[2]*self.conv2_2.shape.as_list()[3], 6)),
                               tf.reshape(loc_hook2, (-1, self.conv3_2.shape.as_list()[1]*self.conv3_2.shape.as_list()[2]*self.conv3_2.shape.as_list()[3], 6)),
                               tf.reshape(loc_hook3, (-1, self.conv4_2.shape.as_list()[1]*self.conv4_2.shape.as_list()[2]*self.conv4_2.shape.as_list()[3], 6))],
                               axis=1)

    # Define cls loss.
    cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_ph_cls, logits=cls_hooks_flat)
    cls_loss = tf.reduce_mean(cls_loss)
    self.cls_loss = cls_loss

    # Define loc loss.
    diff = self.y_ph_loc - loc_hooks_flat
    # loc_loss_L2 = 0.5*(diff**2)
    # loc_loss_L1 = tf.abs(diff) - 0.5
    # smooth_cond = tf.less(tf.abs(diff), 1.0)
    # loc_loss = tf.where(smooth_cond, loc_loss_L1, loc_loss_L2)
    loc_loss = tf.abs(diff)

    ia_cast = tf.expand_dims(tf.cast(self.y_ph_cls[...,1], tf.float32), -1)
    ia_dup = tf.tile(ia_cast, [1,1,6])
    loc_loss = tf.reduce_mean(tf.multiply(loc_loss, ia_dup))
    self.loc_loss = loc_loss
    # Combine losses linearly.
    self.loss = cls_loss + loc_loss_lambda * loc_loss

    # Define optimizer.
    self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)

  def probe(self, X, shape, probe_path):
    """
    Args:
      X (np.ndarray): array of pointclouds (batches, num_points, 3)
    """
    pcs = []
    problem_pcs = []
    counter = 0
    #probe_npy = np.load(probe_path, dtype=np.float32, mmap_mode='w+', shape=())
    probe_memmap = np.memmap(probe_path, dtype='float32', mode='w+', shape=(len(X), self.probe_steps, 
                             self.probe_steps, self.probe_steps, self.num_kernels, self.probes_per_kernel))
    for pc in X:

      process = psutil.Process(os.getpid())
      if process.memory_info().rss // 1e9 > 63.0:
        print("Memory cap surpassed. Exiting...")
        exit()

      ## ADD MEMMAP STUFF HERE
      pc = np.array([pc])
      counter += 1

      #if counter not in [211, 302, 328, 779, 785, 922, 940] and (counter >922 or counter ==1):
      # if counter >= 211 or counter == 1:
      #if counter not in [302, 328, 779, 785, 922, 940]:
      if counter not in [75, 325, 395, 407, 408]:
        pc_disc = self.sess.run(self.probe_op, feed_dict={self.points_ph: pc})
      else:
        problem_pcs.append(counter-1)
      
      # pcs.append(pc_disc)
      probe_memmap[counter-1] = pc_disc[0]

      
      if counter % 1 == 0:
        print('Finished probing {} pointclouds'.format(counter))
      
    self.probe_output = probe_memmap

    probe_memmap.flush()

    # pcs = np.array(pcs)
    # pcs = np.squeeze(pcs, axis=1)
    # np.save(probe_path, probe_memmap)
    return probe_memmap, problem_pcs

  def train_val(self, X_trn=None, y_trn_cls=None, y_trn_loc=None, X_val=None, y_val_cls=None, 
                y_val_loc=None, epochs=10, batch_size=4, display_step=100, save_interval=100):

    assert y_trn_cls is not None and y_trn_loc is not None, "Labels must be defined for train_val call."

    for epoch in range(epochs):
      indices = list(range(X_trn.shape[0]))
      shuffle(indices)
      # X_trn = X_trn[indices]
      # y_trn_cls = y_trn_cls[indices]
      # y_trn_loc = y_trn_loc[indices]

      for step in range(0, X_trn.shape[0], batch_size): 
        randomized_indices = indices[step:step+batch_size]
        # batch_x = X_trn[step:step+batch_size]
        # batch_y_cls = y_trn_cls[step:step+batch_size]
        # batch_y_loc = y_trn_loc[step:step+batch_size]
        batch_x = X_trn[randomized_indices]
        batch_y_cls = y_trn_cls[randomized_indices]
        batch_y_loc = y_trn_loc[randomized_indices]
        _, loss, cl, ll = self.sess.run([self.optimizer, self.loss, self.cls_loss, self.loc_loss], feed_dict={self.X_ph: batch_x, 
                                            self.y_ph_cls: batch_y_cls, self.y_ph_loc: batch_y_loc})

        if step % display_step == 0:
          print("Epoch: {}, Iter: {}, Loss: {:.6f}.".format(epoch, step, loss))

      if X_val is not None and y_val_cls is not None and y_val_loc is not None:
        val_loss = 0
        for step in range(0, X_val.shape[0], batch_size):
          val_batch_x = X_val[step:step+batch_size]
          val_batch_y_cls = y_val_cls[step:step+batch_size]
          val_batch_y_loc = y_val_loc[step:step+batch_size]
          val_loss += self.sess.run(self.loss, 
                      feed_dict={self.X_ph: val_batch_x, self.y_ph_cls: val_batch_y_cls, self.y_ph_loc: val_batch_y_loc})

        print("Epoch: {}, Validation Loss: {:6f}.".format(epoch, 
                                                       val_loss*batch_size/X_val.shape[0]))
      if epoch % save_interval == 0 and self.ckpt_save is not None:
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

  def save_checkpoint(self, checkpoint_dir, step, name='ssnn_model'):
    if not isdir(checkpoint_dir):
      makedirs(checkpoint_dir)
    print("Saving model checkpoint to {}.".format(checkpoint_dir))
    self.saver.save(self.sess, join(checkpoint_dir, name), global_step=step)

  def load_checkpoint(self, checkpoint_dir, name='ssnn_model-25'):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, name))
      return True
    return False
