from tf_models.base.model import BaseModel
import tensorflow as tf
from tqdm import tqdm
import numpy as np

class Model(object):
    def __init__(self, options):
        self.options = options
        
    def conv2d(self, input, ks, stride):
        w = tf.get_variable('weights', shape=ks, initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', shape=[ks[-1]], initializer=tf.constant_initializer())
        out = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME')
        out = tf.nn.bias_add(out, b)
        return out

    def make_conv_bn_relu(self, input, ks, stride, is_training):
        out = self.conv2d(input, ks, stride)
        out = tf.layers.batch_normalization(out, training=is_training)
        out = tf.nn.relu(out, name=tf.get_variable_scope().name)
        return out

    def make_fc(self, input, ks, keep_prob):
        w = tf.get_variable('weights', shape=ks, initializer=tf.truncated_normal_initializer())
        b = tf.get_variable('biases', shape=[ks[-1]], initializer=tf.constant_initializer())
        out = tf.matmul(input, w)
        out = tf.nn.bias_add(out, b, name=tf.get_variable_scope().name)
        # out = tf.nn.dropout(out, keep_prob, name='%s-drop' % name)
        return out

    def forward(self, img, is_training, keep_prob):
        with tf.variable_scope('conv1') as scope:
            out = self.conv2d(img, [3, 3, 3, 16], 2)
            # out = tf.layers.batch_normalization(out, name='bn1', training=is_training)
            out = tf.nn.relu(out, name=scope.name)

        with tf.variable_scope('conv2'):
            out = self.make_conv_bn_relu(out, [3, 3, 16, 64], 1, is_training)
            out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')
            
        with tf.variable_scope('conv3'):
            out = self.make_conv_bn_relu(out, [5, 5, 64, 128], 1, is_training)
            out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('conv4'):
            out = self.make_conv_bn_relu(out, [7, 7, 128, 256], 1, is_training)
            out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        with tf.variable_scope('conv5'):
            out = self.make_conv_bn_relu(out, [9, 9, 256, 512], 1, is_training)
            out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        out = tf.reshape(out, [-1, 512 * 20 * 23])
        with tf.variable_scope('fc6'):
            out = self.make_fc('fc1', out, [512 * 20 * 23, 512], keep_prob)

        with tf.variable_scope('fc7'):
            out = self.make_fc(out, [512, 2], keep_prob)
        return out

    def build_model(self, is_training, inputs):
        #self.x = tf.placeholder(tf.float32, shape=[None, 640, 720, 3])
        #self.y = tf.placeholder(tf.float32, shape=[None, 2])
        self.keep_prob = tf.placeholder(tf.float32)
        
        self.pred = self.forward(inputs.x, self.is_training, self.keep_prob)
        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.pred - inputs.y), 1)))
        optimizer = tf.train.AdamOptimizer(self.options.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_op = optimizer.minimize(self.loss)

    def predict(self, x):
        predict_y = self.sess.run(self.pred, feed_dict={
            self.x: x,
            self.is_training: False,
            self.keep_prob: 1
        })
        return predict_y[0]

    def train_step(self, x, y):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.is_training: True,
            self.keep_prob: self.options.keep_prob
        }
        _, loss = self.sess.run([self.train_op, self.loss], feed_dict=feed_dict)
        return loss

    def score(self, x, y):
        feed_dict = {
            self.x: x,
            self.y: y,
            self.is_training: False,
            self.keep_prob: 1.0
        }
        loss = self.sess.run([self.loss], feed_dict=feed_dict)
        return loss

    def tower_loss(self, scope, x, y):
        pred = self.forward(self.x, self.is_training, self.keep_prob)
        loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(pred - y), 1)))
        return loss
    
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

    def train_step_multi_gpu(self):
        with tf.Graph().as_default(), tf.device('/cpu:0'):
            optimizer = tf.train.AdamOptimizer(self.options.learning_rate)
            tower_grads = []
            with tf.variable_scope(tf.get_variable_scope()):
                for i in xrange(self.options.num_gpus):
                    with tf.device('/gpu:%d' % i):
                        with tf.name_scope('tower_%d' % i) as scope:
                            x, y = next(data_set.next_batch(self.options.batch_size))
                            loss = tower_loss(scope, x, y)
                            tf.get_variable_scope().reuse_variables()
                            grads = optimizer.compute_gradients(loss)
                            tower_grads.append(grads)
                grads = average_gradients(tower_grads)
                train_op = optimizer.apply_gradients(grads)
            for cur_epoch in range(self.options.num_epochs)
                loop = tqdm(range(self.options.num_iter_per_epoch))
                losses = []
                for it in loop:
                    _, loss_value = self.sess.run([train_op, loss])
                    losses.append(loss_value)
                    avg_loss = np.mean(losses)
                    print('loss={}'.format(avg_loss))
                    self.save()
