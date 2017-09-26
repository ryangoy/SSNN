import tensorflow as tf
probe_3d_grad_module = tf.load_op_library('build/libprobe3d_grad.so')
tf_probe3d_grad = probe_3d_grad_module.probe_3d_grad

@ops.RegisterGradient("Probe3D")
def _probe_3d_grad_cc(op, grad):
	"""
	The gradients for probe_3d if we can implement them in C++.
	"""
	return tf_probe3d_grad(grad, op.inputs[0]. op.inputs[1])

def _probe_3d_grad_py(op, grad):
	"""
	The gradients for probe_3d if we can implement them in python.
	"""
	return None