import tensorflow as tf 
import numpy as np
import vgg19 as vgg

class DescriptorLoss:
    def __init__(self, images, texture, batch_size=1, content=None):
        self.batch_size = batch_size
        self.model = vgg.Vgg19()
        self.texture = texture
        self.images = images
        self.model.build(self.images)
        if content != None:
            with tf.name_scope('content_model'):
                self.content_model = vgg.Vgg19()
                self.content_model.build(content)
        
    def __gram_tensor(self, tensor):
        with tf.name_scope('gram_tensor'):
            shape = tf.shape(tensor)
            flattened = tf.reshape(tensor,[shape[0], shape[1]*shape[2], shape[3]])
            flattened_t = tf.transpose(flattened, perm=[0,2,1])
            out = tf.batch_matmul(flattened_t, flattened)
            return out

    def __gram_np(self, arr):
        with tf.name_scope('gram_np'):
            flattened = arr.reshape((arr.shape[0], arr.shape[1]*arr.shape[2], arr.shape[3]))
            flattened_t = np.transpose(flattened, (0,2,1))
            out = np.matmul(flattened_t,flattened)
            return out

    def __accumulate_loss(self, x, y, fn, lossCoeff):
        with tf.name_scope('acc_loss'):
            out = sum([lossCoeff(target) * tf.reduce_sum(tf.square(fn(tensor, target))) for tensor, target in zip(x,y)])
            return out

    def texture_loss(self, session):    
        model = self.model
        with tf.name_scope('texture_loss'):
            textures = np.asarray([self.texture]*self.batch_size)
            texture_fetch = [model.conv1_1, model.conv2_1, model.conv3_1, model.conv4_1, model.conv5_1]        
            feed_dict = { self.images: textures }        
            texture_response = session.run(texture_fetch, feed_dict=feed_dict)
#            texture_response[2] = texture_response[2] * .15
            return self.__accumulate_loss(texture_fetch, texture_response, 
                lambda x, y: self.__gram_tensor(x) - self.__gram_np(y), 
                lambda target: 1. / (4 * target.shape[0] * (target.shape[1] * target.shape[2])**2 * target.shape[3]**2))

    def content_loss_tensor(self, session):
        with tf.name_scope('content_loss'):
            return 0.5 * tf.reduce_sum(tf.square(self.model.conv4_2 - self.content_model.conv4_2)) / tf.to_float(tf.shape(self.model.conv4_2)[0])

    def content_loss(self, session, targets):
        model = self.model
        with tf.name_scope('content_loss'):
            target_fetch = [model.conv4_2]
            feed_dict = { self.images: targets }
            target_response = session.run(target_fetch, feed_dict=feed_dict)

            return self.__accumulate_loss(target_fetch, target_response, 
                lambda x, y: x - y, lambda x: 0.5)

    def build(self, session, targets, alpha):
        self.LT = self.texture_loss(session)
        self.LC = self.content_loss_tensor(session)
#        self.LC = self.content_loss(session, targets)
        return alpha * self.LT + self.LC

    def run(self, session, image):
        session.run(self.images.assign(image))
        

