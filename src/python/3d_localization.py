import tensorflow as tf
import numpy as np

FLAGS = tf.app.flags.FLAGS

FLAGS.DEFINE_string('data_dir', None, 'Path to base directory.')
FLAGS.DEFINE_integer('num_epochs', 1, 'Number of epochs to train.')
# ... 

def main(_):
  return

if __name__ == '__main__':
  tf.app.run()
