import tensorflow as tf
import numpy as np


probe_module = tf.load_op_library('./probe.so')

def probe3d(inp, dims, steps=None, num_kernels=8, probes_per_kernel=16, kernel_size=None, strides=None, name='probe3D'):

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

    # Initialize weights with given parameters.
    weights = tf.Variable(tf.truncated_normal(shape=[num_kernels, probes_per_kernel, 3], stddev=kernel_size[0]/2), name='probe3D')
    return probe_module.probe(inp, weights, dims, steps=steps[0])

def dot_product(inp, stddev=0.1, name='dot_product'):
    """
    This layer weights the output of probe3D. This is what the network trains.
    """
    print inp.shape
    assert len(inp.shape) == 6, "Dot product expects input of shape (batches, kernels, probes_per_kernel, x, y, z)"
    print inp.shape
    # Initialize weights
    weights = tf.Variable(tf.truncated_normal(shape=inp.shape[1:3], stddev=stddev), name=name)
    print weights.shape
    # Dot product over input probe and samples per probe and weights
    axes = tf.constant([[1, 2], [0, 1]])

    # Hooray for Tensorflow methods :)
    dot_product = tf.tensordot(inp, weights, axes=axes)
    new_shape = [-1] + inp.shape[1:2].as_list() + inp.shape[3:].as_list()
    dot_product = tf.reshape(dot_product, new_shape)
    return dot_product


