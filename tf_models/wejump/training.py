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

def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
        Note that this function provides a synchronization point across all towers.
        Args:
            tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
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

def train_multi_gpu(options, inputs):
    with tf.device('/cpu:0'):
        global_step = tf.train.get_or_create_global_step()
        optimizer = tf.train.AdamOptimizer(options.learning_rate)
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in range(options.num_gpus):
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
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
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
            
