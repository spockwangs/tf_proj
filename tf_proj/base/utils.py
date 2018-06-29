import tensorflow as tf

def variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.
    Args:
        name: name of the variable
        shape: list of ints
        initializer: initializer for Variable
    Returns:
        Variable Tensor
    """
    with tf.device('/cpu:0'):
        var = tf.get_variable(name, shape, initializer=initializer)
    return var

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual towers. individual gradients. The inner list is over individual
        gradients for each tower.

        Returns:
            List of pairs of (gradient, variable) where the gradient has been averaged
        across all towers.
        """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)
            
            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def conv2d(input, ks, stride):
    """Make a convolution layer.
    Args:
       input: input features
       ks: kernel tensor of shape [filter_width, filter_height, num_input_channels, num_output_channels]
       stride: the convolution stride
    Returns:
      The convolution output tensor.
    """
    w = variable_on_cpu('weights', ks, tf.truncated_normal_initializer())
    b = variable_on_cpu('biases', [ks[-1]], tf.constant_initializer())
    out = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME')
    out = tf.nn.bias_add(out, b)
    return out

def make_conv_bn_relu(input, ks, stride, is_training):
    out = conv2d(input, ks, stride)
    out = tf.layers.batch_normalization(out, training=is_training)
    out = tf.nn.relu(out, name=tf.get_variable_scope().name)
    return out

def make_fc(input, ks, keep_prob):
    w = variable_on_cpu('weights', ks, tf.truncated_normal_initializer())
    b = variable_on_cpu('biases', [ks[-1]], tf.constant_initializer())
    out = tf.matmul(input, w)
    out = tf.nn.bias_add(out, b, name=tf.get_variable_scope().name)
    return out
