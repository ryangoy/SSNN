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

    return output, weights

def dot_product(inputs, filters=1, stddev=0.01, name='dot_product'):
    """
    This layer weights the output of probe3D. This is the first layer the network trains.
    """
    assert len(inputs.shape) == 7, "Dot product expects input of shape (batches, x, y, z, kernels, probes_per_kernel, 4)"
    # Weight dims are similar to convolutional weights:
    #   Convolutional weights are (kernel_width, kernel_height, num_input_features, num_output_features)
    #   Dot product weights are (kernels, probes_per_kernel, num_output_features)
    weights = tf.Variable(tf.truncated_normal(shape=inputs.shape[-3:].as_list() + [filters], stddev=stddev), name=name)
    
    # Hooray for Tensorflow methods :)
    dot_product = tf.tensordot(inputs, weights, axes=3)
  
    # Output shape should be (batches, x, y, z, output_features)
    new_shape = [-1] + inputs.shape[1:4].as_list() + [filters]
    dot_product = tf.reshape(dot_product, new_shape)
    return dot_product, weights



def backprojection(inp, d_img, K, RT, filters):

    # output = tf.placeholder(tf.float32, (None, 32, 32, 32, filters))

    tf_func = tf.py_func(project_2d_to_3d, [inp, d_img, K, RT], tf.float32)
    tf_func.set_shape((None, 32, 32, 32, filters))
    return tf_func

def project_2d_to_3d(conv_out, d_img, K, RT):

    step_x = int(d_img.shape[1]/conv_out.shape[1])
    step_y = int(d_img.shape[2]/conv_out.shape[2])
    index_matrix = np.stack(np.meshgrid(np.arange(1, d_img.shape[1]+1, step=step_x), np.arange(1, d_img.shape[2]+1), step=step_y), axis=-1).reshape((-1, 2))

    # processing of depth map
    d_img = np.right_shift(d_img, 3)
    d_img = d_img[:, ::step_x,::step_y].astype(float).flatten()/1000
    feature_vec = conv_out.reshape((-1, 3))

    # apply inverse K matrix

    cx = K[:,0,2]
    cy = K[:,1,2]
    fx = K[:,0,0]
    fy = K[:,1,1]

    x = (index_matrix[...,0]-cx) * d_img / fx
    y = (index_matrix[...,1]-cy) * d_img / fy
    z = d_img

    # join channels
    x = np.expand_dims(x, -1)
    y = np.expand_dims(y, -1)
    z = np.expand_dims(z, -1)
    KiX = np.concatenate([x, z, -y], axis=-1)

    # apply rotation and translation from extrinsics
    R = RT[:,:, :3]
    T = RT[:,:, 3:]
    Y = KiX - np.tile(T, (1, 1, KiX.shape[0])).T
    Y = Y.dot(np.linalg.inv(R))

    output_tensor = np.zeros((conv_out.shape[0], conv_out.shape[1], conv_out.shape[2], conv_out.shape[2], conv_out.shape[3]))
    output_tensor[Y] = feature_vec

    return output_tensor
