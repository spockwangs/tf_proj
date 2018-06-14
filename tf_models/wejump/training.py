#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#

import tensorflow as tf
import numpy as np
from tqdm import tqdm
from .model_fn import build_model

def train(options, model, inputs):
    saver = tf.train.Saver(max_to_keep=options.max_to_keep)
    global_step = tf.train.get_global_step()
    with tf.Session() as sess:
        sess.run([model['global_variable_init_op'], inputs['iterator_init_op']])
        latest_checkpoint = tf.train.latest_checkpoint(options.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            print("Model loaded")
        for epoch in range(options.num_epochs):
            loop = tqdm(range(options.num_iter_per_epoch))
            losses = []
            for it in loop:
                _, loss = sess.run([model['train_op'], model['loss']])
                losses.append(loss)
            avg_loss = np.mean(losses)
            print('loss={}'.format(avg_loss))
            print('Saving model ...')
            saver.save(sess, options.checkpoint_dir, global_step=global_step)
            
