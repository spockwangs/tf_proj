#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#

import tensorflow as tf
import tf_proj.base.utils as utils

def conv2d(input, ks, stride):
    w = utils.variable_on_cpu('weights', ks, tf.truncated_normal_initializer())
    b = utils.variable_on_cpu('biases', [ks[-1]], tf.constant_initializer())
    out = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME')
    out = tf.nn.bias_add(out, b)
    return out

def make_conv_bn_relu(input, ks, stride, is_training):
    out = conv2d(input, ks, stride)
    out = tf.layers.batch_normalization(out, training=is_training)
    out = tf.nn.relu(out, name=tf.get_variable_scope().name)
    return out

def make_fc(input, ks, keep_prob):
    w = utils.variable_on_cpu('weights', ks, tf.truncated_normal_initializer())
    b = utils.variable_on_cpu('biases', [ks[-1]], tf.constant_initializer())
    out = tf.matmul(input, w)
    out = tf.nn.bias_add(out, b, name=tf.get_variable_scope().name)
    return out

def inference(options, features, is_training):
    """Build a model.
    Args:
        options (dict): hyper parameters
        features: An tf.Tensor of shape [?, 640, 720, 3]
        is_training: training mode or evaluation mode.

    Returns:
        An label.
    """
    if is_training:
        keep_prob = options.keep_prob
    else:
        keep_prob = 1.0
    with tf.variable_scope('conv1') as scope:
        out = conv2d(features, [3, 3, 3, 16], 2)
        # out = tf.layers.batch_normalization(out, name='bn1', training=is_training)
        out = tf.nn.relu(out, name=scope.name)
        out_tmp = tf.transpose(out, perm=[3, 1, 2, 0])[:, :, :, 0:1]
        tf.summary.image('conv1', out_tmp)

    with tf.variable_scope('conv2'):
        out = make_conv_bn_relu(out, [3, 3, 16, 64], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        out_tmp = tf.transpose(out, perm=[3, 1, 2, 0])[:, :, :, 0:1]
        tf.summary.image('conv2', out_tmp)
        
    with tf.variable_scope('conv3'):
        out = make_conv_bn_relu(out, [5, 5, 32, 128], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        out_tmp = tf.transpose(out, perm=[3, 1, 2, 0])[:, :, :, 0:1]
        tf.summary.image('conv3', out_tmp)
        
    with tf.variable_scope('conv4'):
        out = make_conv_bn_relu(out, [7, 7, 64, 256], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        out_tmp = tf.transpose(out, perm=[3, 1, 2, 0])[:, :, :, 0:1]
        tf.summary.image('conv4', out_tmp)

    with tf.variable_scope('conv5'):
        out = make_conv_bn_relu(out, [9, 9, 128, 512], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        out_tmp = tf.transpose(out, perm=[3, 1, 2, 0])[:, :, :, 0:1]
        tf.summary.image('conv5', out_tmp)
        
    out = tf.reshape(out, [-1, 512 * 20 * 23])
    with tf.variable_scope('fc6'):
        out = make_fc(out, [512 * 20 * 23, 256], keep_prob)

    with tf.variable_scope('fc7'):
        out = make_fc(out, [256, 2], keep_prob)
    return out

def compute_loss(predictions, labels):
    '''Compute loss.

    Args:
        predictions: the predictions from inference().
        labels: labels from the inputs.

    Returns:
        Loss tensor.
    '''
    return tf.reduce_mean(tf.sqrt(tf.pow(predictions - labels, 2) + 1e-12))
    #return tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(predictions - labels), 1)))

def get_train_op_and_loss(options, features, labels, global_step):
    """Get train op and loss.
    Args:
        options (dict): An object which contains the hyper-parameters.
        features (tf.Tensor)
        labels (tf.Tensor)
        global_step (tf.Variable)
    Returns:
        train_op: Train opertation.
        loss: loss tensor.
    """
    with tf.device('/cpu:0'):
        lr = tf.train.exponential_decay(
            options.learning_rate, global_step, options.decay_steps, options.decay_rate, staircase=True)
        tf.summary.scalar('lr', lr)
        optimizer = tf.train.AdamOptimizer(lr)
        if len(options.gpus) == 0:
            predict = inference(options, features, is_training=True)
            loss = compute_loss(predict, labels)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(loss, global_step=global_step)
        elif len(options.gpus) == 1:
            with tf.device('/gpu:0'):
                predict = inference(options, features, is_training=True)
                loss = compute_loss(predict, labels)
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(loss, global_step=global_step)
        else:
            tower_grads = []
            losses = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(len(options.gpus)):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_%d' % i) as scope:
                            predict = inference(options, features, is_training=True)
                            tower_loss = compute_loss(predict, labels)
                            losses.append(tower_loss)
                            tf.get_variable_scope().reuse_variables()
                            grads = optimizer.compute_gradients(tower_loss)
                            tower_grads.append(grads)
            loss = tf.reduce_mean(losses)
            grads = utils.average_gradients(tower_grads)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(grads, global_step=global_step)
    return train_op, loss
