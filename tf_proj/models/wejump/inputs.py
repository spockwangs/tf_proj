#!/usr/bin/env python3
#-*- coding: utf-8 -*-
#
# @author spockwang@tencent.com
#

import random
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

def get_data_batch(name):
    name = name.decode()
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
    img = img[320:-320, :, :].astype(np.float32)
    label = np.array([x-320, y], dtype=np.float32)
    return img, label

def _generator(name_list):
    for name in name_list:
        x, y = get_data_batch(name)
        print('name={}, y={}'.format(name, y))
        yield x, y

def get_train_inputs(options):
    """Returns train inputs.
    Args:
        options: dict
    Returns:
        features: 4-D tf.Tensor features of shape [batch_size, 640, 720, 3].
        labels: 2-D tf.Tensor labels of shape [batch_size, 2].
    """
    name_list = get_name_list(options.data_dir)
    name_list = name_list[200:]
    dataset = tf.data.Dataset.from_tensor_slices(name_list)
    dataset = dataset.shuffle(buffer_size=len(name_list)).map(lambda filename: tuple(tf.py_func(get_data_batch, [filename], [tf.float32, tf.float32])))
    num = len(options.gpus)
    if num <= 0:
        num = 1
    dataset = dataset.repeat().batch(options.batch_size)
    iter = dataset.make_one_shot_iterator()
    x, y = iter.get_next()
    return x, y
    
def get_train_inputs2(options):
    """Returns train inputs.
    Args:
        options: dict
    Returns:
        features: 4-D tf.Tensor features of shape [batch_size, 640, 720, 3].
        labels: 2-D tf.Tensor labels of shape [batch_size, 2].
    """
    name_list = get_name_list(options.data_dir)
    name_list = name_list[200:]
    name_list = np.random.choice(name_list, options.batch_size)
    for idx, name in enumerate(name_list):
        img, label = get_data_batch(name)
        if idx == 0:
            images = img[np.newaxis, :, :, :]
            labels = label[np.newaxis, :]
        else:
            img = img[np.newaxis, :, :, :]
            label = label[np.newaxis, :]
            images = np.concatenate((images, img), axis=0)
            labels = np.concatenate((labels, label), axis=0)
    return images, labels

def get_eval_inputs(options):
    '''
    Args:
        is_trainig: bool
        options: dict
    Returns:
        features: 4-D tf.Tensor features of shape [batch_size, 640, 720, 3].
        labels: 2-D tf.Tensor labels of shape [batch_size, 2].
    '''
    def _generator_eval(name_list):
        for name in name_list:
            x, y = get_data_batch(name.e)
            yield x, y

    name_list = get_name_list(options.data_dir)
    name_list = name_list[:200]
    dataset = tf.data.Dataset.from_tensor_slices(name_list)
    dataset = dataset.map(lambda filename: tuple(tf.py_func(get_data_batch, [filename], [tf.float32, tf.float32])))
    dataset = dataset.batch(options.batch_size).prefetch(options.batch_size)
    iter = dataset.make_one_shot_iterator()
    x, y = iter.get_next()
    return x, y

if __name__ == '__main__':
    flags = tf.app.flags
    flags.DEFINE_string('config', 'tf_proj/models/wejump/config.json', '')
    FLAGS = flags.FLAGS
    
    from tf_proj.base.options import get_options
    options = get_options(FLAGS.config)
    x, y = get_train_inputs(options)
    with tf.Session() as sess:
        x_val, y_val = sess.run([x, y])
        print(x_val.shape, y_val.shape)
      
