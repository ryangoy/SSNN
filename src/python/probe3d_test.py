import tensorflow as tf

class Probe3DTest(tf.test.TestCase):
  def testProbe3D(self):
    probe_3D_module = tf.load_op_library('./probe_3d.so')
    with self.test_session():
      result = probe_3D_module.probe_3d([5, 4, 3, 2, 1])
    print(result)

if __name__ == "__main__":
  tf.test.main()
