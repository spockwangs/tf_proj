import tensorflow as tf

def generator_forward(z):
    with tf.variable_scope("generator"):
        h1 = tf.layers.dense(z, 128)
        h1 = tf.maximum(0.01*h1, h1)
        h1 = tf.layers.dropout(h1, rate=0.2)
    
        logits = tf.layers.dense(h1, 784)
        output = tf.sigmoid(logits)
        return output, logits
    
def discriminator_forward(x, reuse=False):
    with tf.variable_scope("discriminator", reuse=reuse):
        h1 = tf.layers.dense(x, 128)
        h1 = tf.maximum(0.01*h1, h1)
        logits = tf.layers.dense(h1, 1)
        output = tf.sigmoid(logits)
        return output, logits
