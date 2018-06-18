#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#


import tensorflow as tf
import numpy as np
import tf_proj.models.wejump.model as model
from .inputs import get_eval_inputs

def evaluate(options):
    features, labels = get_eval_inputs(options)
    predict = model.inference(options, features, is_training=False)
    loss = model.compute_loss(predict, labels)
    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run([init_op])
        latest_checkpoint = tf.train.latest_checkpoint(options.checkpoint_dir)
        if latest_checkpoint:
            print("Loading model checkpoint {} ...".format(latest_checkpoint))
            saver.restore(sess, latest_checkpoint)
            print("Model loaded")
        losses = []
        while True:
            try:
                loss_val = sess.run(loss)
                losses.append(loss_val)
            except tf.errors.OutOfRangeError:
                break
        avg_loss = np.mean(losses)
        print('loss={}'.format(avg_loss))
            
