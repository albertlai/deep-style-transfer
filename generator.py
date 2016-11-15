import tensorflow as tf 
import numpy as np
from layer_utils import get_norm
leak_alpha = 0.001

class GeneratorNetwork:
    def __init__(self, image, is_training, k, inf_noise=0.):
        self.k = k
        self.is_training= is_training
        self.image_inputs = []
        self.noise_inputs = []
        self.image_in = image
        shape = image.get_shape().as_list()
        for i in xrange(0, k):
            if i>0:
                image = tf.nn.avg_pool(self.image_inputs[-1], ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME', name='image%d'%i)
            max_noise = 1. if is_training else inf_noise
            noise = tf.random_uniform(image.get_shape().as_list(), 0., max_noise, dtype = tf.float32, name="noise%d"%i)
            self.noise_inputs.append(noise)
            self.image_inputs.append(image)
        self.build()


    def join_block(self, x, y, name):
        with tf.variable_scope(name):
            x_dim = x.get_shape().as_list()[1:3]
            scaled = tf.image.resize_images(y, x_dim[0], x_dim[1], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            x_norm = get_norm(x)
            y_norm = get_norm(scaled)
            return tf.concat(3, [x_norm, y_norm], name="concat")

    def conv_block(self, bottom, n, S, name, prefix, relu=True):
        with tf.variable_scope(name):
            if n > 1 and self.is_training:
                pad = n//2
                bottom = tf.pad(bottom, [[0,0],[pad,pad],[pad,pad],[0,0]], "REFLECT")
            C = bottom.get_shape().as_list()[-1]
            filt = self.get_conv_filter(n, C, S, name, prefix)
            if self.is_training:
                conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='VALID')
            else:
                conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            conv_biases = self.get_bias(name, S)
            bias = tf.nn.bias_add(conv, conv_biases)
            if relu:
                norm = get_norm(bias)
                return tf.nn.relu(norm)
            else:
                return bias

    def triple_block(self, bottom, S, prefix):
        with tf.variable_scope(prefix):
            a = self.conv_block(bottom, 3, S, "A-3x%d" % S, prefix)
            b = self.conv_block(a, 3, S, "B-3x%d" % S, prefix)
            c = self.conv_block(b, 1, S, "C-1x%d" % S, prefix)
            return c

    def build(self):
        k = self.k
        self.z = {}
        self.layer1 = []
        S = 8
        for i in xrange(k):
            n = 2**i
            name = "z%d_triple_%d" % (i, S)
            with tf.variable_scope("scale-1-%d" % n):
                zi = tf.concat(3, [self.noise_inputs[i], self.image_inputs[i]])
                self.z[i] = zi
            self.layer1.append(self.triple_block(zi, S, name))

        self.joins = [self.layer1[-1]]
        for i in reversed(xrange(k-1)):
            S += 8
            joined = self.join_block(self.layer1[i], self.joins[-1], 'join_%d-%d' % (i, i+1))
            conv = self.triple_block(joined, S, "triple_%d" % S)
            self.joins.append(conv)

        conv = self.conv_block(self.joins[-1], 1, 3, "1x30", "out-", relu=False)        
        self.out = tf.mul(tf.tanh(conv/255.0), 255, name="output")        
        return self.out

    def get_conv_filter(self, n, C, S, name, prefix):
        kernel = tf.Variable(tf.random_normal([n, n, C, S], stddev=np.sqrt(2./(n*n*C)), name="weights"))
        if self.is_training:
            self.variable_summaries(kernel, prefix+name)
        return kernel

    def get_bias(self, name, S):
        biases = tf.Variable(tf.zeros([S]), name="biases")
        return biases

    def variable_summaries(self, var, name, log_histogram=False):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            if log_histogram:
                tf.histogram_summary(name, var)    

