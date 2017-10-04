import tensorflow as tf
import numpy as np

class Probe3DTest(tf.test.TestCase):
  def testProbe3D(self):
    probe_3d_module = tf.load_op_library('./probe_3d.so')
    
    xyzrgb = np.load('/home/ryan/cs/datasets/SSNN/test/input_data.npy')[0]
    print np.array(xyzrgb).shape

    with self.test_session():
      # const T* input, const T* weights, const T* dims, const T* steps
      weights = np.array([[[0, 0, 0]]]).astype(float)
      dims = np.array([10, 10, 10]).astype(float)
      steps = np.array([10, 10 ,10]).astype(float)
      print ('running result')

      result = probe_3d_module.probe3d(np.array([xyzrgb[:,:3]]), weights, dims, steps)
      print result
      print 'finished running'



if __name__ == "__main__":
  tf.test.main()
