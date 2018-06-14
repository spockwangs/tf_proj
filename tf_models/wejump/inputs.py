#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#

import numpy as np
import tensorflow as tf
import cv2
import os

def get_name_list(data_dir):
    name_list = []
    for i in range(3, 10):
        dir = os.path.join(data_dir, 'exp_%02d' % i)
        this_name = os.listdir(dir)
        this_name = [os.path.join(dir, name) for name in this_name]
    name_list = name_list + this_name
    name_list_raw = name_list
    name_list = filter(lambda name: 'res' in name, name_list)
    name_list = list(name_list)

    def _name_checker(name):
        posi = name.index('_res')
        img_name = name[:posi] + '.png'
        if img_name in name_list_raw:
            return True
        else:
            return False

    name_list = list(filter(_name_checker, name_list))
    return name_list

def get_data_batch(batch_name):
    batch = {}
    for idx, name in enumerate(batch_name):
        posi = name.index('_res')
        img_name = name[:posi] + '.png'
        x, y = name[name.index('_h_') + 3: name.index('_h_') + 6], name[name.index('_w_') + 3: name.index('_w_') + 6]
        x, y = int(x), int(y)
        img = cv2.imread(img_name)

        # 将中间的白点去除，用附近的颜色填充。
        mask1 = (img[:, :, 0] == 245)
        mask2 = (img[:, :, 1] == 245)
        mask3 = (img[:, :, 2] == 245)
        mask = mask1 * mask2 * mask3
        img[mask] = img[x + 10, y + 14, :]

        # 截取中间的640*720的区域作为训练特征，同时调整目标点的坐标。
        img = img[320:-320, :, :]
        label = np.array([x-320, y], dtype=np.float32)

        if idx == 0:
            batch['img'] = img[np.newaxis, :, :, :]
            batch['label'] = label.reshape([1, label.shape[0]])
        else:
            img_tmp = img[np.newaxis, :, :, :]
            label_tmp = label.reshape((1, label.shape[0]))
            batch['img'] = np.concatenate((batch['img'], img_tmp), axis=0)
            batch['label'] = np.concatenate((batch['label'], label_tmp), axis=0)
    return batch['img'], batch['label']

def next_batch():
    batch_name = np.random.choice(self.train_name_list, batch_size)
    x, y = get_data_batch(batch_name)
    yield x, y

def test_set():
    x, y = get_data_batch(self.val_name_list)
    return x, y

def inputs(is_training, options):
    '''
    Args:
        is_trainig: bool
        options: dict
    '''
    name_list = get_name_list(options.data_dir)
    def _generator(name_list):
        for name in name_list:
            x, y = get_data_batch([name])
            yield x[0], y[0]
    if is_training:
        dataset = tf.data.Dataset().from_generator(lambda: _generator(name_list[200:]),
                                                   output_types=(tf.float32, tf.float32),
                                                   output_shapes=((640, 720, 3), (2,)))
        dataset = dataset.repeat().batch(options.batch_size).prefetch(options.batch_size)
    else:
        dataset = tf.data.Dataset().from_generator(lambda: _generator(name_list[:200]),
                                                   output_types=(tf.float32, tf.float32),
                                                   output_shapes=((640, 720, 3), (2,)))
        dataset = dataset.batch(options.batch_size).prefetch(options.batch_size)

    iter = dataset.make_initializable_iterator()
    x, y = iter.get_next()
    iterator_init_op = iter.initializer
    inputs = {
        'x': x,
        'y': y,
        'iterator_init_op': iterator_init_op
    }
    return inputs
