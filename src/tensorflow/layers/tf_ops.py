import tensorflow as tf
import numpy as np


probe_module = tf.load_op_library('./probe.so')
dot_product_module = tf.load_op_library('./dot_product.so')

def probe3D(inp, dims, steps=None, num_kernels=8, probes_per_kernel=16, kernel_size=None, strides=None):

    assert type(dims) is np.ndarray, "dims must be of type numpy.ndarray."
    assert steps is not None or strides is not None, "steps or strides must be defined."
    if strides is None:
        strides = dims / steps
    if steps is None:
        steps = dims / strides
    if len(set(steps)) != 1:
        print("Warning: probe3D does not support different sized steps. Only the first dimension will be used.")
    if kernel_size is None:
        kernel_size = 2*strides

    assert type(num_kernels) is int, "num_kernels must be of type int."
    assert type(steps) is list, "steps must be of type list."
    assert type(probes_per_kernel) is int, "probes_per_kernel must be of type int."

    steps = steps

    # Initialize weights with given parameters.
    weights = tf.Variable(tf.truncated_normal(shape=[num_kernels, probes_per_kernel, 3], stddev=kernel_size[0]/2), name='probe3D_weights')
    return probe_module.probe(inp, weights, dims, steps=steps[0])

def dot_product(inp, stddev=0.1):
    assert len(inp.shape) == 6, "Dot product expects input of shape (batches, probes, samples_per_probe, x, y, z)"
    weights = tf.Variable(tf.truncated_normal(shape=inp.shape[1:3], stddev=stddev), name='dot_product_weights')
    return dot_product_module.dot_product(inp, weights)


