#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#


import tensorflow as tf

def _variable_on_cpu(name, shape, initializer):
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

def conv2d(input, ks, stride):
    w = _variable_on_cpu('weights', ks, tf.truncated_normal_initializer())
    b = _variable_on_cpu('biases', [ks[-1]], tf.constant_initializer())
    out = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME')
    out = tf.nn.bias_add(out, b)
    return out

def make_conv_bn_relu(input, ks, stride, is_training):
    out = conv2d(input, ks, stride)
    out = tf.layers.batch_normalization(out, training=is_training)
    out = tf.nn.relu(out, name=tf.get_variable_scope().name)
    return out

def make_fc(input, ks, keep_prob):
    w = _variable_on_cpu('weights', ks, tf.truncated_normal_initializer())
    b = _variable_on_cpu('biases', [ks[-1]], tf.constant_initializer())
    out = tf.matmul(input, w)
    out = tf.nn.bias_add(out, b, name=tf.get_variable_scope().name)
    return out

def inference(img, is_training, keep_prob):
    """Build a model.
    Args:
        img: An tf.Tensor of shape [?, 640, 720]
        is_training: training mode or evaluation mode.
        keep_prob: an float

    Returns:
        An label.
    """
    with tf.variable_scope('conv1') as scope:
        out = conv2d(img, [3, 3, 3, 16], 2)
        # out = tf.layers.batch_normalization(out, name='bn1', training=is_training)
        out = tf.nn.relu(out, name=scope.name)

    with tf.variable_scope('conv2'):
        out = make_conv_bn_relu(out, [3, 3, 16, 64], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            
    with tf.variable_scope('conv3'):
        out = make_conv_bn_relu(out, [5, 5, 64, 128], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv4'):
        out = make_conv_bn_relu(out, [7, 7, 128, 256], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    with tf.variable_scope('conv5'):
        out = make_conv_bn_relu(out, [9, 9, 256, 512], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

    out = tf.reshape(out, [-1, 512 * 20 * 23])
    with tf.variable_scope('fc6'):
        out = make_fc(out, [512 * 20 * 23, 512], keep_prob)

    with tf.variable_scope('fc7'):
        out = make_fc(out, [512, 2], keep_prob)
    return out

def loss(predict_y, labels):
    '''Compute loss.

    Args:
        predict_y: the predictions from inference().
        labels: labels from the inputs.

    Returns:
        Loss tensor.
    '''

    return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(predict_y - labels), 1)))

