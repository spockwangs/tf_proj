#!/usr/bin/env python3
#-*- coding: utf-8 -*-

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from network_gan import generator_forward, discriminator_forward
    
z_dim = 100
z = tf.placeholder(tf.float32, shape=[None, z_dim])
x = tf.placeholder(tf.float32, shape=[None, 784])
gen_image, gen_logits = generator_forward(z)
disc_real, real_logits = discriminator_forward(x)
disc_fake, fake_logits = discriminator_forward(gen_image, reuse=True)
d_real_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=real_logits, labels=tf.ones_like(real_logits)))
d_fake_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.zeros_like(fake_logits)))
d_loss = d_real_loss + d_fake_loss
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=fake_logits, labels=tf.ones_like(fake_logits)))

# 分别定义generator和discriminator的参数，分别优化。
train_vars = tf.trainable_variables()
g_vars = [var for var in train_vars if var.name.startswith("generator")]
d_vars = [var for var in train_vars if var.name.startswith("discriminator")]
optimizer = tf.train.AdamOptimizer(0.001)
disc_train_op = optimizer.minimize(d_loss, var_list=d_vars)
gen_train_op = optimizer.minimize(g_loss, var_list=g_vars)

batch_size = 128
epochs = 500
mnist = tf.contrib.learn.datasets.DATASETS['mnist']('/tmp/mnist')
train_x = np.reshape(mnist.train.images, [-1, 784])
init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    for e in range(epochs):
        for step in range(mnist.train.num_examples//batch_size):
            _, disc_loss, d_real_loss_val, d_fake_loss_val = sess.run([disc_train_op, d_loss, d_real_loss, d_fake_loss],
                                    feed_dict={
                                        x: train_x[np.random.choice(train_x.shape[0], batch_size), :],
                                        z: np.random.uniform(-1, 1, size=(batch_size, z_dim))
                                    })
            _, gen_loss = sess.run([gen_train_op, g_loss],
                                   feed_dict={
                                       z: np.random.uniform(-1, 1, size=(batch_size, z_dim))
                                   })
        print("epoch={}, disc_loss={}(real_loss={}, fake_loss={}), gen_loss={}".format(
            e, disc_loss, d_real_loss_val, d_fake_loss_val, gen_loss))

        if e > 0 and e % 5 == 0:
            n = 10
            digit_size = 28
            figure = np.zeros((digit_size * n, digit_size * n))
            # linearly spaced coordinates corresponding to the 2D plot
            # of digit classes in the latent space
            grid_x = np.linspace(-4, 4, n)
            grid_y = np.linspace(-4, 4, n)[::-1]
            for i, yi in enumerate(grid_y):
                for j, xi in enumerate(grid_x):
                    z_sample = np.random.uniform(-1, 1, size=(1, z_dim))
                    x_decoded = sess.run([gen_image],
                                         feed_dict={
                                             z: z_sample
                                         })
                    digit = x_decoded[0].reshape(digit_size, digit_size)
                    figure[i * digit_size: (i + 1) * digit_size,
                           j * digit_size: (j + 1) * digit_size] = digit

            plt.figure(figsize=(10, 10))
            start_range = digit_size // 2
            end_range = n * digit_size + start_range + 1
            pixel_range = np.arange(start_range, end_range, digit_size)
            sample_range_x = np.round(grid_x, 1)
            sample_range_y = np.round(grid_y, 1)
            plt.xticks(pixel_range, sample_range_x)
            plt.yticks(pixel_range, sample_range_y)
            plt.xlabel("z[0]")
            plt.ylabel("z[1]")
            plt.imshow(figure, cmap='Greys_r')
            plt.savefig('gan_{}.png'.format(e))

