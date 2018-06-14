#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#


import tensorflow as tf

def conv2d(input, ks, stride):
    w = tf.get_variable('weights', shape=ks, initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', shape=[ks[-1]], initializer=tf.constant_initializer())
    out = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME')
    out = tf.nn.bias_add(out, b)
    return out

def make_conv_bn_relu(input, ks, stride, is_training):
    out = conv2d(input, ks, stride)
    out = tf.layers.batch_normalization(out, training=is_training)
    out = tf.nn.relu(out, name=tf.get_variable_scope().name)
    return out

def make_fc(input, ks, keep_prob):
    w = tf.get_variable('weights', shape=ks, initializer=tf.truncated_normal_initializer())
    b = tf.get_variable('biases', shape=[ks[-1]], initializer=tf.constant_initializer())
    out = tf.matmul(input, w)
    out = tf.nn.bias_add(out, b, name=tf.get_variable_scope().name)
    return out

def forward(img, is_training, keep_prob):
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

def build_model(options, x, y, is_training):
    '''Build a model.

    Args:
        options: {object} hyperparmaeters
        x: {tf.Tensor<?, 640, 720>} input features
        y: {tf.Tensor<?, 2>} labels
        is_training: {bool} build model for training or evaluation

    Returns:
        Dict: A dict which contains { 'pred', 'loss', 'train_op', 'global_variable_init_op' }.
    '''

    model = {}
    global_step = tf.train.get_or_create_global_step()
    model['pred'] = forward(x, is_training, options.keep_prob)
    model['loss'] = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(model['pred'] - y), 1)))
    optimizer = tf.train.AdamOptimizer(options.learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        model['train_op'] = optimizer.minimize(model['loss'], global_step=global_step)
    model['global_variable_init_op'] = tf.global_variables_initializer()
    return model
