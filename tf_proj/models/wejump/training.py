#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#

from datetime import datetime
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import tf_proj.base.utils as utils
import tf_proj.models.wejump.model as model
from tf_proj.models.wejump.model2 import model_fn
from .inputs import get_train_inputs, get_train_inputs2

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
        lr = tf.placeholder(tf.float32, shape=(1,), name='lr')
        tf.summary.scalar('lr', lr)
        optimizer = tf.train.AdamOptimizer(options.learning_rate)
        if len(options.gpus) == 0:
            model = model_fn(options, features, labels, mode='train')
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.minimize(model['loss'], global_step=global_step)
                loss = model['loss']
        elif len(options.gpus) == 1:
            with tf.device('/gpu:0'):
                model = model_fn(options, features, labels, mode='train')
                update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(update_ops):
                    train_op = optimizer.minimize(model['loss'], global_step=global_step)
                    loss = model['loss']
        else:
            tower_grads = []
            losses = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in range(len(options.gpus)):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_%d' % i) as scope:
                            model = model_fn(options, features, labels, mode='train')
                            losses.append(model['loss'])
                            tf.get_variable_scope().reuse_variables()
                            grads = optimizer.compute_gradients(tower_loss)
                            tower_grads.append(grads)
            loss = tf.reduce_mean(losses)
            grads = utils.average_gradients(tower_grads)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                train_op = optimizer.apply_gradients(grads, global_step=global_step)
    return train_op, loss

def train2(options):
    global_step = tf.train.get_or_create_global_step()
    features, labels = get_train_inputs(options)
    train_op, loss = get_train_op_and_loss(options, features, labels, global_step)
    saver = tf.train.Saver(max_to_keep=options.max_to_keep)
    tf.summary.scalar('loss', loss)
    
    class _StopAtLoss(tf.train.SessionRunHook):
        def __init__(self, loss):
            self._stop_loss = loss
            
        def begin(self):
            ckpt_path = tf.train.latest_checkpoint(options.checkpoint_dir)
            if ckpt_path:
                restored_step = int(ckpt_path.split('/')[-1].split('-')[-1])
                self._assign_op = tf.assign(global_step, restored_step)
            
        def after_create_session(self, session, coord):
            if hasattr(self, '_assign_op'):
                session.run(self._assign_op)
                print('restored global_step={}'.format(session.run(global_step)))

        def before_run(self, run_context):
            return tf.train.SessionRunArgs([loss, global_step])  # Asks for loss value.

        def after_run(self, run_context, run_values):
            loss_value, step = run_values.results
            if step % 10 == 0:
                format_str = ('%s: step %d, loss = %.2f')
                print(format_str % (datetime.now(), step, loss_value))
                if loss_value < self._stop_loss:
                    run_context.request_stop()

    init_op = tf.global_variables_initializer()
    tf.logging.set_verbosity(tf.logging.INFO)
    with tf.train.MonitoredTrainingSession(
            hooks=[
                _StopAtLoss(1.0),
                tf.train.CheckpointSaverHook(
                    checkpoint_dir=options.checkpoint_dir,
                    save_steps=100,
                    saver=saver,
                    checkpoint_basename="wejump.ckpt"),
                tf.train.SummarySaverHook(
                    save_steps=10,
                    output_dir=options.summary_dir,
                    summary_op=tf.summary.merge_all())
            ],
            config=tf.ConfigProto(allow_soft_placement=True)) as mon_sess:
        ckpt_path = tf.train.latest_checkpoint(options.checkpoint_dir)
        if ckpt_path:
            print("Loading model: {}".format(ckpt_path))
            saver.restore(mon_sess, ckpt_path)    
            print('Model loaded')
        lr = 0.1
        losses_queue = []
        prev_avg_loss = None
        while not mon_sess.should_stop():
            print('lr={}'.format(lr))
            loss, _ = mon_sess.run([loss, train_op], feed_dict={ 'lr': lr })
            losses_queue.append(loss)
            if len(losses_queue) >= 10:
                avg_loss = np.mean(losses_queue)
                if prev_avg_loss is not None and avg_loss >= prev_avg_loss:
                    lr = lr*0.1
                prev_avg_loss = avg_loss
            
        
def train(options):
    """Train the model.
    Args:
        options: An object which contains the hyper-parameters.
    """
    global_step = tf.train.get_or_create_global_step()
    features, labels = get_train_inputs(options)
    train_op, loss = model.get_train_op_and_loss(options, features, labels, global_step)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=options.max_to_keep)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run([init_op])
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


