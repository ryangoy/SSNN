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
from utils import output_to_bboxes, flatten_output
from compute_bbox_accuracy import compute_accuracy
from utils import softmax
from compute_mAP import compute_map

class SSNN:
  
  def __init__(self, dims, num_kernels=1, probes_per_kernel=1, dot_layers=8, probe_steps=32, probe_hook_steps=16, 
               num_scales=3, ckpt_load=None, ckpt_save=None, ckpt_load_iter=50, loc_loss_lambda=1, learning_rate=0.0001, k_size_factor=1,
               num_classes=2, dropout=0.9):

    self.hook_num = 1
    self.dims = dims
    self.probe_steps = probe_steps
    self.probe_size = dims / probe_steps
    self.probe_output = None
    self.ckpt_save = ckpt_save
    self.ckpt_load = ckpt_load
    self.probe_hook_steps = probe_hook_steps
    self.num_kernels = num_kernels
    self.probes_per_kernel = probes_per_kernel
    self.ckpt_load_iter = ckpt_load_iter
    self.dropout=dropout
    self.num_classes=num_classes

    # Defines self.probe_op
    self.init_probe_op(dims, probe_steps, num_kernels=num_kernels, 
                       probes_per_kernel=probes_per_kernel, k_size_factor=k_size_factor)

    # Defines self.X_ph, self.y_ph, self.model, self.cost, self.optimizer
    self.init_model(num_kernels, probes_per_kernel, probe_steps, probe_hook_steps, num_scales, num_classes, dot_layers=dot_layers, loc_loss_lambda=loc_loss_lambda, learning_rate=learning_rate)

    # Initialize variables
    self.saver = tf.train.Saver()
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)
    # self.sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    self.sess = tf.Session()
    init_op = tf.global_variables_initializer()
    self.sess.run(init_op)

    if self.ckpt_load and self.load_checkpoint(self.ckpt_load, iteration=self.ckpt_load_iter):
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
    self.points_ph = tf.placeholder(tf.float32, (None, None, 6))

    # Shape: (batches, x, y, z, probes, samples per probe)
    self.probe_op, self.probe_coords = probe3d(self.points_ph, dims, 
                            steps=steps, 
                            num_kernels=num_kernels, 
                            probes_per_kernel=probes_per_kernel,
                            k_size_factor=k_size_factor)

  def hook_layer(self, input_layer, reuse=False, activation=None, num_classes=2):
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

      input_layer_cls = tf.layers.conv3d(input_layer, filters=64, kernel_size=3, padding='SAME',
                              strides=1, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())



      input_layer_cls = tf.nn.dropout(input_layer_cls, self.dropout)

      #input_layer_cls = tf.contrib.layers.batch_norm(input_layer_cls)
      # Predicts the confidence of whether or not an objects exists per feature.
      conf = tf.layers.conv3d(input_layer_cls, filters=num_classes, kernel_size=1, padding='SAME',
                              strides=1, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())


      input_layer_loc = tf.layers.conv3d(input_layer, filters=64, kernel_size=3, padding='SAME',
                              strides=1, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())

      input_layer_loc = tf.nn.dropout(input_layer_loc, self.dropout)
      # input_layer_loc = tf.contrib.layers.batch_norm(input_layer_loc)

      # Predicts the center coordinate and relative scale of the box
      loc = tf.layers.conv3d(input_layer_loc, filters=6, kernel_size=1, padding='SAME',
                              strides=1, activation=activation, kernel_initializer=tf.contrib.layers.xavier_initializer())

      self.hook_num += 1

      return conf, loc

  def init_model(self, num_kernels, probes_per_kernel, probe_steps, probe_hook_steps, num_scales, num_classes,
                 learning_rate=0.0001, loc_loss_lambda=1, dot_layers=8, reuse_hook=False):

    # Shape: (batches, x_steps, y_steps, z_steps, num_kernels, 
    #         probes_per_kernel)
    self.X_ph = tf.placeholder(tf.float32, (None, probe_steps, probe_steps, 
                                            probe_steps, num_kernels, 
                                            probes_per_kernel, 4))

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
    self.dot_product = tf.nn.relu(self.dot_product)

    self.dot_product = tf.nn.dropout(self.dot_product, self.dropout)

    # self.conv0_1 = tf.layers.conv3d(self.dot_product, filters=64, kernel_size=3, 
    #                   strides=1, padding='SAME', activation=tf.nn.relu, 
    #                   kernel_initializer=tf.contrib.layers.xavier_initializer())
    # # self.conv0_1 = tf.nn.dropout(self.conv0_1, dropout)
    # self.conv0_2 = tf.layers.conv3d(self.conv0_1, filters=64, kernel_size=3, 
    #                   strides=1, padding='SAME', activation=tf.nn.relu, 
    #                   kernel_initializer=tf.contrib.layers.xavier_initializer())

    # self.pool0 = tf.nn.max_pool3d(self.conv0_2, ksize=[1, 2, 2, 2, 1], 
    #                               strides=[1, 2, 2, 2, 1], padding='SAME')
    
    self.conv1_1 = tf.layers.conv3d(self.dot_product, filters=64, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
    # self.conv0_1 = tf.nn.dropout(self.conv0_1, dropout)
    self.conv1_2 = tf.layers.conv3d(self.conv1_1, filters=64, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    self.pool1 = tf.nn.max_pool3d(self.conv1_2, ksize=[1, 2, 2, 2, 1], 
                                  strides=[1, 2, 2, 2, 1], padding='SAME')

    self.conv2_1 = tf.layers.conv3d(self.pool1, filters=64, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())
    # self.conv1_1 = tf.nn.dropout(self.conv1_1, dropout)
    self.conv2_2 = tf.layers.conv3d(self.conv2_1, filters=64, kernel_size=3, 
                      strides=1, padding='SAME', activation=tf.nn.relu, 
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    # First hook layer.
    cls_hook1, loc_hook1 = self.hook_layer(self.conv2_2, reuse=reuse_hook, num_classes=num_classes)

    self.pool2 = tf.nn.max_pool3d(self.conv2_2, ksize=[1, 2, 2, 2, 1], 
                                  strides=[1, 2, 2, 2, 1], padding='SAME')

    self.conv3_1 = tf.layers.conv3d(self.pool2, filters=128, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    self.conv3_2 = tf.layers.conv3d(self.conv3_1 , filters=128, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Second hook layer, resolution is 1/2 the first
    cls_hook2, loc_hook2 = self.hook_layer(self.conv3_2, reuse=reuse_hook, num_classes=num_classes)

    self.pool3 = tf.nn.max_pool3d(self.conv3_2, ksize=[1, 2, 2, 2, 1], 
                                  strides=[1, 2, 2, 2, 1], padding='SAME')

    self.conv4_1 = tf.layers.conv3d(self.pool3, filters=256, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    self.conv4_2 = tf.layers.conv3d(self.conv4_1 , filters=256, kernel_size=3,
                      strides=1, padding='SAME', activation=tf.nn.relu,
                      kernel_initializer=tf.contrib.layers.xavier_initializer())

    # Third hook layer, resolution is 1/4th the first
    cls_hook3, loc_hook3 = self.hook_layer(self.conv4_2, reuse=reuse_hook, num_classes=num_classes)

    self.cls_hooks = [cls_hook1, cls_hook2, cls_hook3]
    self.loc_hooks = [loc_hook1, loc_hook2, loc_hook3]

    self.cls_hooks_flat = tf.concat([tf.reshape(cls_hook1, (-1, self.conv2_2.shape.as_list()[1]*self.conv2_2.shape.as_list()[2]*self.conv2_2.shape.as_list()[3], num_classes)),
                               tf.reshape(cls_hook2, (-1, self.conv3_2.shape.as_list()[1]*self.conv3_2.shape.as_list()[2]*self.conv3_2.shape.as_list()[3], num_classes)),
                               tf.reshape(cls_hook3, (-1, self.conv4_2.shape.as_list()[1]*self.conv4_2.shape.as_list()[2]*self.conv4_2.shape.as_list()[3], num_classes))],
                               axis=1)
    self.loc_hooks_flat = tf.concat([tf.reshape(loc_hook1, (-1, self.conv2_2.shape.as_list()[1]*self.conv2_2.shape.as_list()[2]*self.conv2_2.shape.as_list()[3], 6)),
                               tf.reshape(loc_hook2, (-1, self.conv3_2.shape.as_list()[1]*self.conv3_2.shape.as_list()[2]*self.conv3_2.shape.as_list()[3], 6)),
                               tf.reshape(loc_hook3, (-1, self.conv4_2.shape.as_list()[1]*self.conv4_2.shape.as_list()[2]*self.conv4_2.shape.as_list()[3], 6))],
                               axis=1)


    # Mask out the voxels that don't have a bounding box associated with it. Note that y_ph_cls holds one-hot vectors.
    pos_mask = tf.cast(tf.reduce_sum(self.y_ph_cls[...,1:], axis=-1), tf.float32)
    pos_mask_exp = tf.expand_dims(pos_mask, -1)
    pos_mask_loc = tf.tile(pos_mask_exp, [1,1,6])
    neg_mask = tf.cast(self.y_ph_cls[...,0], tf.float32)
    neg_mask_exp = tf.expand_dims(neg_mask, -1)
    neg_mask_loc = tf.tile(neg_mask_exp, [1,1,6])

    N_pos = tf.reduce_sum(pos_mask)
    N_neg = tf.reduce_sum(neg_mask)

    epsilon = tf.ones_like(self.cls_hooks_flat) * .00001
    logits = tf.add(self.cls_hooks_flat, epsilon)

    # Define cls loss.
    cls_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_ph_cls, logits=logits)
    neg_loss = 500* tf.multiply(neg_mask, cls_loss) / (N_neg + 1)
    pos_loss = tf.multiply(pos_mask, cls_loss) / (N_pos + 1)

    cls_loss = neg_loss + pos_loss 
    cls_loss = tf.reduce_sum(cls_loss)
    
    # Define loc loss.
    loc_loss = tf.square(self.y_ph_loc - self.loc_hooks_flat)
    loc_loss = tf.multiply(loc_loss, pos_mask_loc) / (N_pos + 1)
    loc_loss = tf.reduce_sum(tf.where(tf.is_nan(loc_loss), tf.zeros_like(loc_loss), loc_loss))

    self.cls_loss = cls_loss
    self.loc_loss = loc_loss

    # Combine losses linearly.
    self.loss = cls_loss + loc_loss_lambda * loc_loss

    # Define optimizer.
    self.optimizer = tf.contrib.opt.NadamOptimizer(learning_rate).minimize(self.loss)

  def probe(self, X, shape, probe_path):
    """
    Args:
      X (np.ndarray): array of pointclouds (batches, num_points, 3)
    """
    pcs = []
    problem_pcs = []
    counter = 0
    probe_memmap = np.memmap(probe_path, dtype='float32', mode='w+', shape=(len(X), self.probe_steps, 
                             self.probe_steps, self.probe_steps, self.num_kernels, self.probes_per_kernel, 4))
    for pc in X:

      process = psutil.Process(os.getpid())
      if process.memory_info().rss // 1e9 > 110.0:
        print("[WARRNING] Memory cap surpassed. Flushing to hard disk.")
        probe_memmap.flush()

      # Batch size of 1.
      pc = np.array([pc])
      counter += 1
      #if counter not in [211, 302, 328, 779, 785, 922, 940] and (counter >922 or counter ==1):

      # hack-y way of avoiding problem pointclouds (haven't figured out why this happens)
      #if counter not in [75, 325, 395, 407, 408, 641]: # matterport
      #if counter not in [124]: # stanford
      #if counter not in [140]: # matterport bed
      #if counter not in [117, 218]: # matterport toilet
      #if counter not in [80, 397]: # matterport table
      #if counter not in [225, 444, 445]:  # matterport chair
      #if counter not in [29, 77]: # matterport bathtub
      if counter not in []:
        pc_disc, probe_coords = self.sess.run([self.probe_op, self.probe_coords], feed_dict={self.points_ph: pc})
      else:
        problem_pcs.append(counter-1)
      
      probe_memmap[counter-1] = pc_disc[0]

      if counter % 1 == 0:
        print('\t\tFinished probing {} pointclouds'.format(counter))
      
    self.probe_output = probe_memmap

    probe_memmap.flush()

    # DEBUG
    # np.save('probe_coords.npy', probe_coords)

    return probe_memmap, problem_pcs
    
  def gaussian_noise_jitter(self, input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return tf.add(input_layer, noise)

  def train_val(self, X_trn=None, y_trn_cls=None, y_trn_loc=None, X_val=None, y_val_cls=None, 
                y_val_loc=None, val_bboxes=None, y_val_one_hot=None, epochs=10, batch_size=4, display_step=100, save_interval=100):

    assert y_trn_cls is not None and y_trn_loc is not None, "Labels must be defined for train_val call."

    train_losses = []
    val_losses = []
    mAPs = []
    for epoch in range(epochs):
      indices = list(range(X_trn.shape[0]))

      curr_cl_sum = 0
      curr_ll_sum = 0
      counter = 0
      for step in range(0, X_trn.shape[0], batch_size): 
        shuffle(indices)
        randomized_indices = indices[step:step+batch_size]
        batch_x = X_trn[randomized_indices]
        assert not np.any(np.isnan(batch_x)) # prevent NaNs
        
        # inp = tf.placeholder(tf.float32, shape=batch_x.shape, name='input')
        # noise = self.gaussian_noise_jitter(inp, .003)
        # batch_x = noise.eval(session=self.sess, feed_dict={inp: batch_x})

        batch_y_cls = y_trn_cls[randomized_indices]
        batch_y_loc = y_trn_loc[randomized_indices]
        _, loss, cl, ll = self.sess.run([self.optimizer, self.loss, self.cls_loss, self.loc_loss], 
                                 feed_dict={self.X_ph: batch_x, self.y_ph_cls: batch_y_cls, self.y_ph_loc: batch_y_loc})

        curr_cl_sum += cl
        curr_ll_sum += ll
        counter += 1


        if step % display_step < batch_size and step != 0:
          print("Epoch: {}/{}, Iter: {}, Classification Loss: {:.6f}, Localization Loss: {:.6f}.".format(epoch, epochs, 
                                            step - (step % display_step), 
                                             curr_cl_sum / counter, curr_ll_sum / counter))
          train_losses.append((curr_cl_sum + curr_ll_sum)/counter)
          curr_cl_sum = 0
          curr_ll_sum = 0
          counter = 0

      if X_val is not None and y_val_cls is not None and y_val_loc is not None:
        val_loss = 0
        val_cls_loss = 0
        val_loc_loss = 0
        counter = 0
        val_cls_preds = []
        val_loc_preds = []

        for step in range(0, X_val.shape[0], batch_size):
          val_batch_x = X_val[step:step+batch_size]
          val_batch_y_cls = y_val_cls[step:step+batch_size]
          val_batch_y_loc = y_val_loc[step:step+batch_size]
          vl, vcl, vll, val_cls_pred, val_loc_pred = self.sess.run([self.loss, self.cls_loss, self.loc_loss, self.cls_hooks_flat, self.loc_hooks_flat],
                      feed_dict={self.X_ph: val_batch_x, self.y_ph_cls: val_batch_y_cls, self.y_ph_loc: val_batch_y_loc})

          val_cls_preds.append(val_cls_pred)
          val_loc_preds.append(val_loc_pred)
          val_loss += vl
          val_cls_loss += vcl
          val_loc_loss += vll
          counter += 1

        val_cls_preds = np.concatenate(val_cls_preds, axis=0)
        val_loc_preds = np.concatenate(val_loc_preds, axis=0)
        val_cls_preds = np.apply_along_axis(softmax, 2, val_cls_preds)
        val_bbox_preds, val_cls= output_to_bboxes(val_cls_preds, val_loc_preds, 16, 3, 
                     self.dims/self.probe_hook_steps, None, None, conf_threshold=0)
        mAP_orig = compute_map(val_bbox_preds, val_cls, val_bboxes, y_val_one_hot, use_nms=False)
        mAP = compute_map(val_bbox_preds, val_cls, val_bboxes, y_val_one_hot)
        print("Epoch: {}/{}, Validation Classification Loss: {:.6f}, Localization Loss: {:.6f}, mAP: {:.6f}.".format(epoch, epochs,
                                                       val_cls_loss / counter, val_loc_loss / counter, mAP))
        val_losses.append((val_cls_loss + val_loc_loss)/counter)
        mAPs.append(mAP)

      if epoch != 0 and (epoch % save_interval == 0 or epoch == epochs-1) and self.ckpt_save is not None:
        self.save_checkpoint(self.ckpt_save, epoch)
    np.save('mAPs.npy', np.array(mAPs))
    np.save('val_losses.npy', np.array(val_losses))
    np.save('train_losses.npy', np.array(train_losses))

  def test(self, X_test, save_dir=None, batch_size=1):
    cls_preds = []
    loc_preds = []
    for i in range(0, X_test.shape[0], batch_size):
      batch_x = X_test[i:i+batch_size]
      hooks, dp_weights = self.sess.run([self.cls_hooks + self.loc_hooks, self.dp_weights], feed_dict={self.X_ph: batch_x})
      cls_preds.append(hooks[:3])
      loc_preds.append(hooks[3:])

      # DEBUG
      # np.save('dp_weights.npy', dp_weights)

    return cls_preds, loc_preds

  def save_checkpoint(self, checkpoint_dir, step, name='ssnn_model'):
    if not isdir(checkpoint_dir):
      makedirs(checkpoint_dir)
    print("Saving model checkpoint to {}.".format(checkpoint_dir))
    self.saver.save(self.sess, join(checkpoint_dir, name), global_step=step)

  def load_checkpoint(self, checkpoint_dir, name='ssnn_model-', iteration=50):
    ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
    if ckpt and ckpt.model_checkpoint_path:
      self.saver.restore(self.sess, os.path.join(checkpoint_dir, name+str(iteration)))
      return True
    return False
