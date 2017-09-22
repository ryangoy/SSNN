import numpy as np
import tensorflow as tf
# import sc_layer

class SSNN:
  
  def __init__(self, input_dims):

    # Defines self.x, self.y, self.model, self.cost, self.optimizer
    self.init_model(input_dims, step_size)
 
  def max_pool(self, x, kshape, name='conv2d'):
    """
    Args:
      x: an input tensor
      kshape: a tuple with shape [batch_step, width_step, height_step, feature_step]
      name: name of layer
    """
    return tf.nn.max_pool(x, ksize=kshape, strides=kshape, 
                          padding='SAME', name=name)

  def fc_layer(self, x, num_nodes, name='fc_layer', activation=tf.nn.relu):
    with tf.variable_scope(name, reuse=False):
      W = tf.Variable(shape=[x.shape[-1], num_nodes], 
                      initializer=tf.contrib.layers.xavier_initializer())
      b = tf.Variable(tf.zeros(shape=[num_nodes]))
      return tf.nn.bias_add(tf.matmul(x, W), b)

  def dropout(self, x, keep_prob):
    return tf.nn.dropout(x, keep_prob)

  def init_model(self, input_dims, step_size, learning_rate=0.01):
    """
    Args:
      input_dims: tuple [x_meters, y_meters, z_meters]
      step_size: tuple [x_stride, y_stride, z_stride]
    """

    self.X = tf.placeholder(tf.float32, (None,) + input_shape)
    self.y = tf.placeholder(tf.float32, [None, num_classes])
    
    ### TO DO ###
    

    # Linear activation.
    self.model = self.fc_layer(outputs[-1], num_classes)

    # Probability error for each class, which is assumed to be independent.
    self.cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=self.y, logits=self.model))
    self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.cost)

  def train_val(self, X_trn, y_trn, X_val=None, y_val=None, epochs=10, batch_size=1,
                display_step=100):
    init = tf.global_variables_initializer()

    with tf.Session() as sess:
      sess.run(init)
      for epoch in range(epochs):
        for step in range(int(X_trn.shape[0]/batch_size)):
          batch_x, batch_y = self.get_next_batch(X_trn, y_trn, batch_size)
          sess.run(self.optimizer, feed_dict={x: batch_x, y: batch_y})

          if step % display_step == 0:
            loss, acc = sess.run([self.cost, self.accuracy], feed_dict={x: batch_x, y: batch_y})
            print("Iter {}, Minibatch Loss={:.6f}, Training Accuracy={:.5f}.".format(step, loss, acc))

          
        if X_val is not None and y_val is not None:
          loss = sess.run(self.cost, feed_dict={x: X_val, y: y_val})

        print("Epoch {}, Validation Loss={:6f}, Validation Accuracy={:.5f}.".format(loss, acc))



if __name__ == '__main__':
  TSANet((64, 512, 512, 1), 16)
