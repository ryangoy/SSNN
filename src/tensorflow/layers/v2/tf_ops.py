import tensorflow as tf
import numpy as np


probe_module = tf.load_op_library('./probe.so')

def probe3D(inp, dims, steps=None, num_kernels=8, probes_per_kernel=16, kernel_size=None, strides=None):

    assert steps is not None or strides is not None, "Steps or strides must be defined."
    if strides is None:
        strides = dims / steps
    if steps is None:
        steps = dims / strides
    if len(set(steps)) != 0:
        print("Warning: probe3D does not support different sized steps. Only the first dimension will be used.")
    if kernel_size is None:
        kernel_size = 2*strides

    steps = steps

    # Initialize weights with given parameters.
    weights = tf.Variable(tf.truncated_normal(shape=[num_kernels, probes_per_kernel, 3], stddev=kernel_size[0]/2), name='probe3D_weights')
    return probe_module.probe(inp, weights, dims, steps=steps[0])


