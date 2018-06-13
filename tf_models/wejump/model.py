from tf_models.base.model import BaseModel
import tensorflow as tf

class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        
    def conv2d(self, name, input, ks, stride):
        with tf.name_scope(name):
            with tf.variable_scope(name):
                w = tf.get_variable('%s-w' % name, shape=ks, initializer=tf.truncated_normal_initializer())
                b = tf.get_variable('%s-b' % name, shape=[ks[-1]], initializer=tf.constant_initializer())
                out = tf.nn.conv2d(input, w, strides=[1, stride, stride, 1], padding='SAME', name='%s-conv' % name)
                out = tf.nn.bias_add(out, b, name='%s-biad_add' % name)
        return out

    def make_conv_bn_relu(self, name, input, ks, stride, is_training):
        out = self.conv2d('%s-conv' % name, input, ks, stride)
        out = tf.layers.batch_normalization(out, name='%s-bn' % name, training=is_training)
        out = tf.nn.relu(out, name='%s-relu' % name)
        return out

    def make_fc(self, name, input, ks, keep_prob):
        with tf.name_scope(name):
            with tf.variable_scope(name):
                w = tf.get_variable('%s-w' % name, shape=ks, initializer=tf.truncated_normal_initializer())
                b = tf.get_variable('%s-b' % name, shape=[ks[-1]], initializer=tf.constant_initializer())
                out = tf.matmul(input, w, name='%s-mat' % name)
                out = tf.nn.bias_add(out, b, name='%s-bias_add' % name)
                # out = tf.nn.dropout(out, keep_prob, name='%s-drop' % name)
        return out

    def forward(self, img, is_training, keep_prob, name='fine'):
        with tf.name_scope(name):
            with tf.variable_scope(name):
                out = self.conv2d('conv1', img, [3, 3, 3, 16], 2)
                # out = tf.layers.batch_normalization(out, name='bn1', training=is_training)
                out = tf.nn.relu(out, name='relu1')

                out = self.make_conv_bn_relu('conv2', out, [3, 3, 16, 64], 1, is_training)
                out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

                out = self.make_conv_bn_relu('conv3', out, [5, 5, 64, 128], 1, is_training)
                out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

                out = self.make_conv_bn_relu('conv4', out, [7, 7, 128, 256], 1, is_training)
                out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

                out = self.make_conv_bn_relu('conv5', out, [9, 9, 256, 512], 1, is_training)
                out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

                out = tf.reshape(out, [-1, 512 * 20 * 23])
                out = self.make_fc('fc1', out, [512 * 20 * 23, 512], keep_prob)
                out = self.make_fc('fc2', out, [512, 2], keep_prob)
        return out

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None, 640, 720, 3])
        self.y = tf.placeholder(tf.float32, shape=[None, 2])
        self.keep_prob = tf.placeholder(tf.float32)

        self.pred = self.forward(self.x, self.is_training, self.keep_prob)
        self.loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(self.pred - self.y), 1)))
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
