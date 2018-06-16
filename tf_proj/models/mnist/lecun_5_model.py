from tf_models.base.model import BaseModel
import tensorflow as tf

class Lecun5Model(BaseModel):
    def __init__(self, config):
        super(Lecun5Model, self).__init__(config)
        self.build_model()
        
    def forward(self, img, is_training, keep_prob, name='fine'):
        # padding
        #img = tf.pad(tf.reshape(img, [-1,28,28,1]),[[0,0],[2,2],[2,2],[0,0]])

        # c1
        filter_kernel = tf.Variable(tf.truncated_normal([5, 5, 1, 6], stddev=0.1))
        out = tf.nn.conv2d(img, filter_kernel, strides=[1, 1, 1, 1], padding="SAME", name="c1-conv")
        bias = tf.Variable(tf.zeros([6]))
        out = tf.nn.bias_add(out, bias)

        # s2
        out = tf.nn.avg_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="s2")
        
        # c3
        filter_kernel = tf.Variable(tf.truncated_normal([5, 5, 6, 16], stddev=0.1))
        out = tf.nn.conv2d(out, filter_kernel, strides=[1, 1, 1, 1], padding="VALID", name="c3-conv")
        bias = tf.Variable(tf.constant(1.0, shape=[16]))
        out = tf.nn.bias_add(out, bias)

        # s4
        out = tf.nn.avg_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="s4")
        
        # c5
        out = tf.reshape(out, [-1, 16*5*5])
        w = tf.Variable(tf.truncated_normal([16*5*5, 120], stddev=0.1))
        out = tf.matmul(out, w)
        b = tf.Variable(tf.constant(1.0, shape=[120]))
        out = tf.nn.bias_add(out, b)
        out = tf.nn.sigmoid(out)

        # f6
        w = tf.Variable(tf.truncated_normal([120, 84], stddev=0.1))
        b = tf.Variable(tf.constant(1.0, shape=[84]))
        out = tf.matmul(out, w)
        out = tf.nn.bias_add(out, b)
        out = tf.nn.sigmoid(out)

        # output
        w = tf.Variable(tf.truncated_normal([84, 10], stddev=0.1))
        b = tf.Variable(tf.constant(1.0, shape=[10]))
        out = tf.matmul(out, w)
        out = tf.nn.bias_add(out, b)
        return out

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)
        self.x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
        self.y = tf.placeholder(tf.int32, shape=[None, 10])
        
        pred = self.forward(self.x, self.is_training, self.options.keep_prob)
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y,logits=pred))
        optimizer = tf.train.GradientDescentOptimizer(self.options.learning_rate)
        self.train_step = optimizer.minimize(self.loss)
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')

        
