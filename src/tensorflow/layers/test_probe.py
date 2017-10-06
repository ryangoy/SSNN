import tensorflow as tf
import numpy as np

# class ProbeTest(tf.test.TestCase):
#   def testProbe(self):
#     probe_module = tf.load_op_library('./probe.so')
    
#     xyzrgb = np.load('/home/ryan/cs/datasets/SSNN/test/input_data.npy')[0]
#     print np.array(xyzrgb).shape

#     with self.test_session():
#       # const T* input, const T* weights, const T* dims, const T* steps
#       weights = np.array([[[0, 0, 0]]]).astype(float)
#       dims = np.array([10, 10, 10]).astype(float)
#       steps = np.array([10, 10 ,10]).astype(float)

#       print ('running result')

#       result = probe_module.probe(np.array([xyzrgb[:,:3]]), weights, dims, steps)
#       print result.eval()
#       print 'finished running'

def test():
  probe_module = tf.load_op_library('./probe.so')
    
  xyzrgb = np.load('/home/ryan/cs/datasets/SSNN/test/input_data.npy')[0]
  print np.array(xyzrgb).shape

  with tf.device('/gpu:0'):
    sess = tf.Session()
    weights = tf.constant(np.array([[[0, 0, 0]]]).astype(np.float32))
    dims = tf.constant(np.array([10, 10, 10]).astype(np.float32))
    steps = tf.constant(np.array([10, 10 ,10]).astype(np.float32))

    ph = tf.placeholder(tf.float32)

    graph = probe_module.probe(ph, weights, dims, steps)

    output = sess.run(graph, feed_dict={ph: xyzrgb[:,:3]})

    print output.shape

if __name__ == "__main__":
  # tf.test.main()
  test()
