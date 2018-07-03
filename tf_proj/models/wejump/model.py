#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#

import tensorflow as tf
import tf_proj.base.utils as utils

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
        out = utils.conv2d(features, [5, 5, 3, 16], 2)
        # out = tf.layers.batch_normalization(out, name='bn1', training=is_training)
        out = tf.nn.relu(out, name=scope.name)
        #out_tmp = tf.transpose(out, perm=[3, 1, 2, 0])[:, :, :, 0:1]
        #tf.summary.image('conv1', out_tmp)

    with tf.variable_scope('conv2'):
        out = utils.make_conv_bn_relu(out, [5, 5, 16, 32], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        #out_tmp = tf.transpose(out, perm=[3, 1, 2, 0])[:, :, :, 0:1]
        #tf.summary.image('conv2', out_tmp)
        
    with tf.variable_scope('conv3'):
        out = utils.make_conv_bn_relu(out, [5, 5, 32, 64], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        #out_tmp = tf.transpose(out, perm=[3, 1, 2, 0])[:, :, :, 0:1]
        #tf.summary.image('conv3', out_tmp)

    with tf.variable_scope('conv4'):
        out = utils.make_conv_bn_relu(out, [3, 3, 64, 128], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        #out_tmp = tf.transpose(out, perm=[3, 1, 2, 0])[:, :, :, 0:1]
        #tf.summary.image('conv4', out_tmp)

    with tf.variable_scope('conv5'):
        out = utils.make_conv_bn_relu(out, [2, 2, 128, 512], 1, is_training)
        out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
        #out_tmp = tf.transpose(out, perm=[3, 1, 2, 0])[:, :, :, 0:1]
        #tf.summary.image('conv5', out_tmp)
        
    out = tf.reshape(out, [-1, 512 * 20 * 23])
    with tf.variable_scope('fc6'):
        out = utils.make_fc(out, [512 * 20 * 23, 512], keep_prob)

    with tf.variable_scope('fc7'):
        out = utils.make_fc(out, [512, 2], keep_prob)
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

