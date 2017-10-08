import tensorflow as tf
import numpy as np
from tf_ops import probe3D
import time

def test():
  probe_module = tf.load_op_library('./probe.so')
    
  xyzrgb = np.load('/home/ryan/cs/datasets/SSNN/test/input_data.npy')[0]

  xyz = xyzrgb[:, :3]
  mins = xyz.min(axis=0)
  maxes = xyz.max(axis=0)
  dims = maxes-mins
  xyz = np.array([xyz-mins])[:10]

  steps = [1, 1, 1]
  num_kernels = 4

  with tf.device('/gpu:0'):
    ph = tf.placeholder(tf.float32)
    graph = probe3D(ph, dims, steps=steps, num_kernels=num_kernels, kernel_size=None, strides=None)
    
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    start = time.time()
    output = sess.run(graph, feed_dict={ph: xyz})
    print 'running op took {} seconds'.format(time.time()-start)
    print output.shape
    print output[0, 0, 0, 0, 0]
    print output.mean()

if __name__ == "__main__":
  # tf.test.main()
  test()
