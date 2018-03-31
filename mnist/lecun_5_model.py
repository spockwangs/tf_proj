from ..base.model import BaseModel
import tensorflow as tf

class Lecun5Model(BaseModel):
    def __init__(self, config):
        super(Lecun5Model, self).__init__(config)
        self.build_model()
        
    def forward(self, img, is_training, keep_prob, name='fine'):
        with tf.name_scope(name):
            with tf.variable_scope(name):
                # padding
                img = tf.pad(tf.reshape(img, [-1,28,28,1]),[[0,0],[2,2],[2,2],[0,0]])

                # c1
                filter_kernel = tf.get_variable('c1-kernel', shape=[5, 5, 1, 6],
                                                initializer=tf.truncated_normal_initializer())
                out = tf.nn.conv2d(img, filter_kernel, strides=[1, 1, 1, 1], padding="VALID", name="c1-conv")
                # s2
                out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID", name="s2")

                # c3
                filter_kernel = tf.get_variable('c3-kernel', shape=[5, 5, 6, 16],
                                                initializer=tf.truncated_normal_initializer())
                out = tf.nn.conv2d(out, filter_kernel, strides=[1, 1, 1, 1], padding="VALID", name="c3-conv")
                # s4
                out = tf.nn.max_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding="VALID", name="s4")

                # c5
                filter_kernel = tf.get_variable('c5-kernel', shape=[5, 5, 16, 120],
                                                initializer=tf.truncated_normal_initializer())
                out = tf.nn.conv2d(out, filter_kernel, strides=[1, 1, 1, 1], padding="VALID", name="c5-conv")

                # f6
                w = tf.get_variable('f6-w', shape=[120, 84], initializer=tf.truncated_normal_initializer())
                b = tf.get_variable('f6-b', shape=[84], initializer=tf.constant_initializer())
                out = tf.reshape(out, [-1, 120])
                out = tf.matmul(out, w)
                out = tf.nn.bias_add(out, b)

                # output
                w = tf.get_variable('output-w', shape=[84, 10], initializer=tf.truncated_normal_initializer())
                b = tf.get_variable('output-b', shape=[10], initializer=tf.constant_initializer())
                out = tf.matmul(out, w)
                out = tf.nn.bias_add(out, b)
                out = tf.nn.softmax(tf.nn.dropout(out,keep_prob))
        return out

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.int32, shape=[None])
        
        y_ = tf.one_hot(self.y, 10, axis=1)
        print(y_.shape)
        pred = self.forward(self.x, self.is_training, self.options.keep_prob)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_,logits=pred))
        optimizer = tf.train.AdamOptimizer(self.options.learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.train_step = optimizer.minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.cast(self.y, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        
