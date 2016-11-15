import tensorflow as tf 
import numpy as np
from layer_utils import get_norm

class ResidualGenerator:
    def __init__(self, image, is_training, deconv=False):
        self.is_training = is_training
        self.image_in = image
        conv_1 = self.conv_block(image, 9, 32, 1, "conv_1")
        conv_2 = self.conv_block(conv_1, 3, 64, 2, "conv_2")
        conv_3 = self.conv_block(conv_2, 3, 128, 2, "conv_3")
        res_1 = self.get_residual(conv_3, "res_1")
        res_2 = self.get_residual(res_1, "res_2")
        res_3 = self.get_residual(res_2, "res_3")
        res_4 = self.get_residual(res_3, "res_4")
        res_5 = self.get_residual(res_4, "res_5")
        if deconv:
            deconv_1 = self.deconv_block(res_5, 3, 64, 2, "deconv_1")
            deconv_2 = self.deconv_block(deconv_1, 3, 32, 2, "deconv_2")
        else:
            deconv_1 = self.resize_conv_block(res_5, 3, 64, "deconv_1")
            deconv_2 = self.resize_conv_block(deconv_1, 3, 32, "deconv_2")            
        conv_out = self.conv_block(deconv_2, 9, 3, 1, "conv_out", relu=False)
        self.out = tf.mul(tf.tanh(conv_out/255.0), 255, name="output")   

    def conv_block(self, bottom, n, S, stride, name, relu=True):
        """ conv -> bias -> (norm -> relu | nop) -> out """
        with tf.variable_scope(name):
            if n > 1 and self.is_training:
                pad = n//2
                bottom = tf.pad(bottom, [[0,0],[pad,pad],[pad,pad],[0,0]], "REFLECT")
            C = bottom.get_shape().as_list()[-1]
            filt = self.get_conv_filter(n, C, S, name)
            if self.is_training:
                conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='VALID')
            else:
                conv = tf.nn.conv2d(bottom, filt, [1, stride, stride, 1], padding='SAME')
            conv_biases = self.get_bias(name, S)            
            bias = tf.nn.bias_add(conv, conv_biases)
            if relu:
                norm = get_norm(bias)
                return tf.nn.relu(norm)
            else:
                return bias
    
    def deconv_block(self, bottom, n, S, stride, name):
        """ AKA convolutional transpose """        
        with tf.variable_scope(name):
            b, h, w, C = bottom.get_shape().as_list()
            filt = self.get_conv_filter(n, S, C, name)
            deconv = tf.nn.conv2d_transpose(bottom, filt, [b, h * stride, w * stride, S], [1, stride, stride, 1], padding='SAME')
            biases = self.get_bias(name, S)            
            bias = tf.nn.bias_add(deconv, biases)
            norm = get_norm(bias)
            return tf.nn.relu(norm)


    def resize_conv_block(self, bottom, n, S, name):        
        """ resize the image then run a conv block on it """        
        with tf.variable_scope(name):
            _, h, w, _ = bottom.get_shape().as_list()
            resized = tf.image.resize_images(bottom, h*2, w*2, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            return self.conv_block(resized, n, S, 1, name)

    def get_conv_filter(self, n, C, S, name):
        kernel = tf.Variable(tf.random_normal([n, n, C, S], stddev=np.sqrt(2./(n*n*C)), name="weights"))
        if self.is_training:
            self.variable_summaries(kernel, name)
        return kernel

    def get_bias(self, name, S):
        biases = tf.Variable(tf.zeros([S]), name="biases")
        return biases

    def get_residual(self, bottom, name, S=3):
        with tf.variable_scope(name):
            _,_,_, C = bottom.get_shape().as_list()
            conv = self.conv_block(bottom, S, C, 1, "%s-a" % name)
            return bottom + get_norm(self.conv_block(conv, S, C, 1, "%s-b" % name, relu=False))

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

