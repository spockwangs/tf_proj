from tf_models.base.model import BaseModel
import tensorflow as tf

class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.build_model()
        
    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None, 784])
        self.y = tf.placeholder(tf.int32, shape=[None])

        W = tf.Variable(tf.zeros([784, 10]))
        b = tf.Variable(tf.zeros([10]))
        y_predict = tf.matmul(self.x, W) + b

        self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=y_predict)
        self.train_step = tf.train.GradientDescentOptimizer(self.options.learning_rate).minimize(self.loss, name='train_step')
        correct_prediction = tf.equal(tf.argmax(y_predict, 1), tf.cast(self.y, tf.int64))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

