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
from .inputs import get_train_inputs

def train2(options):
    global_step = tf.train.get_or_create_global_step()
    features, labels, inputs_init_op = get_train_inputs(options)
    train_op, loss = model.get_train_op_and_loss(options, features, labels, global_step)
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
            if self._assign_op:
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
        while not mon_sess.should_stop():
            mon_sess.run(train_op)
        
def train(options):
    """Train the model.
    Args:
        options: An object which contains the hyper-parameters.
    """
    global_step = tf.train.get_or_create_global_step()
    features, labels, inputs_init_op = get_train_inputs(options)
    train_op, loss = model.get_train_op_and_loss(options, features, labels, global_step)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(max_to_keep=options.max_to_keep)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
        sess.run([inputs_init_op, init_op])
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


