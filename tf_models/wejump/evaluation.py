#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#


import tensorflow as tf
import tf_models.wejump.model_fn as model_fn
import numpy as np

def evaluate(options, inputs):
    predict = model_fn.inference(inputs['x'], False, 1.0)
    loss = model_fn.loss(predict, inputs['y'])
    saver = tf.train.Saver()
    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run([inputs['iterator_init_op'], init_op])
        latest_checkpoint = tf.train.latest_checkpoint(options.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            print("Model loaded")
        while True:
            losses = []
            try:
                loss = sess.run(loss)
                losses.append(loss)
            except tf.errors.OutOfRangeError:
                pass
            avg_loss = np.mean(losses)
            print('loss={}'.format(avg_loss))
            break
            
