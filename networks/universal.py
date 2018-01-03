import tensorflow as tf

def random_variable_init(shape, stddev=0.1):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial)

def const_variable_init(shape, const_value=0.1):
    initial = tf.constant(const_value, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W, strides=[1,2,2,1]):
    return tf.nn.conv2d(x, W, strides=strides, padding='VALID')

def max_pooling(x, ksize=[1,2,2,1], strides=[1,2,2,1]):
    return tf.nn.max_pool(x, ksize, strides, padding='VALID')