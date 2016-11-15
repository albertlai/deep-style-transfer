import tensorflow as tf 


def get_norm(x):
    with tf.variable_scope("instance_norm"):
        eps = 1e-6
        mean, var = tf.nn.moments(x, [1, 2], keep_dims=True)
#        mean = tf.reshape(mean, [-1])
#        var = tf.reshape(var, [-1])
        return (x - mean) / (tf.sqrt(var) + eps)
#        b,h,w,c = x.get_shape().as_list()
#        reshaped = tf.transpose(tf.reshape(tf.transpose(x, [0, 3, 1, 2]), [b*c, h*w]))
#        reshaped = (reshaped-mean) / (tf.sqrt(var) + eps)
#        return tf.transpose(tf.reshape(tf.transpose(reshaped), [b, c, h, w]), [0, 2, 3, 1])