import tensorflow as tf
import numpy as np

# Load probe operations.
probe_module = tf.load_op_library('../cpp/probe.so')

def probe3d(inp, dims, steps=None, num_kernels=8, probes_per_kernel=16, kernel_size=None, strides=None, name='probe3D',
            k_size_factor=3):
    """
    Initializes weights and runs the probing operation by calling the backend Tensorflow code.
    """
    print("Initializing probe op with {} kernels and {} probes per kernel.".format(num_kernels, probes_per_kernel))
    assert type(dims) is np.ndarray, "dims must be of type numpy.ndarray."
    assert steps is not None or strides is not None, "steps or strides must be defined."
    if strides is None:
        strides = dims / steps
    if steps is None:
        steps = dims / strides
    if len(set(steps)) != 1:
        print("Warning: probe3D does not support different sized steps. Only the first dimension will be used.")
    if kernel_size is None:
        kernel_size = strides * k_size_factor

    assert k_size_factor == 3 or k_size_factor == 1, "k_size_factor must be either 1 or 3."

    assert type(num_kernels) is int, "num_kernels must be of type int."
    assert type(steps) is list, "steps must be of type list."
    assert type(probes_per_kernel) is int, "probes_per_kernel must be of type int."

    minval = -1 * (kernel_size - strides) / 2
    maxval = (kernel_size + strides) / 2

    # Initialize weights with given parameters.
    weights = tf.Variable(tf.random_uniform(shape=[num_kernels, probes_per_kernel, 3], minval=minval, maxval=maxval), name='probe3D')
    #weights = tf.Variable(tf.truncated_normal(shape=[num_kernels, probes_per_kernel, 3], mean = (minval+maxval)/2, stddev=(maxval-minval)/2), name='probe3D')
    output = probe_module.probe(inp, weights, xdim=dims[0], ydim=dims[1], zdim=dims[2], steps=steps[0], ksize=kernel_size[0])
    return output

def dot_product(inputs, filters=1, stddev=0.01, name='dot_product'):
    """
    This layer weights the output of probe3D. This is the first layer the network trains.
    """
    assert len(inputs.shape) == 6, "Dot product expects input of shape (batches, x, y, z, kernels, probes_per_kernel)"
    # Weight dims are similar to convolutional weights:
    #   Convolutional weights are (kernel_width, kernel_height, num_input_features, num_output_features)
    #   Dot product weights are (probes, num_input_features, num_output_features)
    weights = tf.Variable(tf.truncated_normal(shape=inputs.shape[-2:].as_list() + [filters], stddev=stddev), name=name)
    
    # Hooray for Tensorflow methods :)
    dot_product = tf.tensordot(inputs, weights, axes=2)
  
    # Output shape should be (batches, x, y, z, output_features)
    new_shape = [-1] + inputs.shape[1:4].as_list() + [filters]
    dot_product = tf.reshape(dot_product, new_shape)
    return dot_product, weights
