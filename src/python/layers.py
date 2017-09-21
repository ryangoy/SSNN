import tensorflow as tf

def probe3d(x, W, b, name='probe3d'):
  """
  Compute feedback signal from probing an input point cloud.
  Args:
    x (Tensor): Input point cloud tensor.
    W (Variable): Probe weights.
    b (Variable): Probe biases.
  """
  def get_sample_coords(W, i, j, k):
    # tf.gather_nd takes in 
    sampled_points = tf.gather_nd(x, W)
  tf.while_loop(  
  
  
  return tf.bias_add(sampled_points, b)
