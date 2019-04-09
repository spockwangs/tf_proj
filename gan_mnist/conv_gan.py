import tensorflow as tf

def generator_forward(z):
    with tf.variable_scope("generator"):
        batch_size = tf.shape(z)[0]
        
        h1 = tf.layers.dense(z, 16*5*5)
        h1 = tf.maximum(0.01*h1, h1)
        h1 = tf.layers.dropout(h1, rate=0.2)

        out = tf.reshape(h1, [-1, 5, 5, 16])

        # out: 10x10@16
        filter_kernel = tf.get_variable('f1', initializer=tf.truncated_normal([2, 2, 16, 16], stddev=0.1))
        out = tf.nn.conv2d_transpose(out, filter_kernel, output_shape=[batch_size, 10, 10, 16],
                                     strides=[1, 2, 2, 1], padding="SAME", name="c1")
        bias = tf.get_variable('b1', initializer=tf.constant(1.0, shape=[16]))
        out = tf.nn.bias_add(out, bias)
        
        # out: 14x14@6
        filter_kernel = tf.get_variable('f2', initializer=tf.truncated_normal([5, 5, 6, 16], stddev=0.1))
        out = tf.nn.conv2d_transpose(out, filter_kernel, output_shape=[batch_size, 14, 14, 6],
                                     strides=[1, 1, 1, 1], padding="VALID", name="c2")
        bias = tf.get_variable('b2', initializer=tf.constant(1.0, shape=[6]))
        out = tf.nn.bias_add(out, bias)
        
        # out: 28x28@1
        filter_kernel = tf.get_variable('f3', initializer=tf.truncated_normal([2, 2, 1, 6], stddev=0.1))
        out = tf.nn.conv2d_transpose(out, filter_kernel, output_shape=[batch_size, 28, 28, 1],
                                     strides=[1, 2, 2, 1], padding="SAME", name="c3")
        bias = tf.get_variable('b3', initializer=tf.constant(1.0, shape=[1]))
        out = tf.nn.bias_add(out, bias)
        
        out = tf.reshape(out, [-1, 784])
        logits = tf.sigmoid(out)
        return out, logits
    
def discriminator_forward(x, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        img = tf.reshape(x, [-1, 28, 28, 1])
        # c1
        # out: 28x28@6
        filter_kernel = tf.get_variable(name='f1',
                                        initializer=tf.truncated_normal([5, 5, 1, 6], stddev=0.1))
        out = tf.nn.conv2d(img, filter_kernel, strides=[1, 1, 1, 1], padding="SAME", name="c1-conv")
        bias = tf.get_variable('b1', initializer=tf.zeros([6]))
        out = tf.nn.bias_add(out, bias)

        # s2
        # out: 14x14@6
        out = tf.nn.avg_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="s2")
        
        # c3
        # out: 10x10@16
        filter_kernel = tf.get_variable('f2', initializer=tf.truncated_normal([5, 5, 6, 16], stddev=0.1))
        out = tf.nn.conv2d(out, filter_kernel, strides=[1, 1, 1, 1], padding="VALID", name="c3-conv")
        bias = tf.get_variable('b2', initializer=tf.constant(1.0, shape=[16]))
        out = tf.nn.bias_add(out, bias)

        # s4
        # out: 5x5@16
        out = tf.nn.avg_pool(out, [1, 2, 2, 1], [1, 2, 2, 1], padding="SAME", name="s4")
        
        # c5
        out = tf.reshape(out, [-1, 16*5*5])
        w = tf.get_variable('f3', initializer=tf.truncated_normal([16*5*5, 120], stddev=0.1))
        out = tf.matmul(out, w)
        b = tf.get_variable('b3', initializer=tf.constant(1.0, shape=[120]))
        out = tf.nn.bias_add(out, b)
        out = tf.nn.sigmoid(out)

        # f6
        w = tf.get_variable('f4', initializer=tf.truncated_normal([120, 84], stddev=0.1))
        b = tf.get_variable('b4', initializer=tf.constant(1.0, shape=[84]))
        out = tf.matmul(out, w)
        out = tf.nn.bias_add(out, b)
        out = tf.nn.sigmoid(out)

        # output
        w = tf.get_variable('w5', initializer=tf.truncated_normal([84, 1], stddev=0.1))
        b = tf.get_variable('b5', initializer=tf.constant(1.0, shape=[1]))
        out = tf.matmul(out, w)
        out = tf.nn.bias_add(out, b)
        logits = out
        out = tf.nn.sigmoid(logits)
        return out, logits
