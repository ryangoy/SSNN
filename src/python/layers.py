import tensorflow as tf

def probe3d(x, W, b, name='probe3d'):
  """
  Compute feedback signal from probing an input point cloud.
  Args:
    x (Tensor): Input point cloud tensor.
    W (Variable): Probe weights.
    b (Variable): Probe biases.
  """
  def tf_knn(x, coords):
    tf.py_func(knn, [x, coords], tf.float32)

  def get_sample_coords(x, W, strides):
    """
    Python code equivalent:
    for filter in W:
      for i in range(0, x.shape[0], strides[0]):
        for j in range(0, x.shape[1], strides[1]):
          for k in range(0, x.shape[2], strides[2]):
            offset = [i, j, k]
            coords.append(filter + offset)
    signals = knn(x, coords)
    return signals.reshape(W.shape/strides.shape)
    """
    # tf.gather_nd takes in 

    tf.while_loop(
  
  
  return tf.bias_add(sampled_points, b)
