#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#

import tensorflow as tf
import numpy as np
from tqdm import tqdm
import tf_models.wejump.model_fn as model_fn

def train(options, inputs):
    """Train the model.
    Args:
        options: An object which contains the hyper-parameters.
        inputs: inputs containing the features, labels and the input pipeline init op.
    """
    global_step = tf.train.get_or_create_global_step()
    predict = model_fn.inference(inputs['x'], True, options.keep_prob)
    loss = model_fn.loss(predict, inputs['y'])
    optimizer = tf.train.AdamOptimizer(options.learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        train_op = optimizer.minimize(loss, global_step=global_step)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=options.max_to_keep)
    with tf.Session() as sess:
        sess.run([inputs['iterator_init_op'], init_op])
        latest_checkpoint = tf.train.latest_checkpoint(options.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            print("Model loaded")
        for epoch in range(options.num_epochs):
            loop = tqdm(range(options.num_iter_per_epoch))
            losses = []
            for it in loop:
                _, loss_value = sess.run([train_op, loss])
                losses.append(loss_value)
            avg_loss = np.mean(losses)
            print('loss={}'.format(avg_loss))
            print('Saving model ...')
            saver.save(sess, options.checkpoint_dir, global_step=global_step)

def train_multi_gpu(options, inputs):
    with tf.device('/cpu:0'):
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(options.learning_rate)
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(options.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('tower_%d' % i) as scope:
                        predict = model_fn.inference(inputs['x'], True, options.keep_prob)
                        loss = model_fn.loss(predict, inputs['y'])
                        tf.get_variable_scope().reuse_variables()
                        grads = optimizer.compute_gradients(loss)
                        tower_grads.append(grads)
            grads = average_gradients(tower_grads)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(grads, global_step=global_step)
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(max_to_keep=options.max_to_keep)
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True)
        sess.run([inputs['iterator_init_op'], init_op])
        latest_checkpoint = tf.train.latest_checkpoint(options.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            print("Model loaded")
        for epoch in range(options.num_epochs):
            loop = tqdm(range(options.num_iter_per_epoch))
            losses = []
            for it in loop:
                _, loss_value = sess.run([train_op, model['loss']])
                losses.append(loss_value)
            avg_loss = np.mean(losses)
            print('loss={}'.format(avg_loss))
            print('Saving model ...')
            saver.save(sess, options.checkpoint_dir, global_step=global_step)
            
