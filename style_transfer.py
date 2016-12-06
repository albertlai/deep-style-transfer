from generator import GeneratorNetwork
from residual_generator import ResidualGenerator
from descriptor_loss import DescriptorLoss

import tensorflow as tf
import numpy as np
import utils
import os
import time

NUM_ITERATIONS = 20
MULTISCALE = "MULTISCALE"
RESIDUAL = "RESIDUAL"

class StyleTransfer:
    """ This encapsulates both the generator network and the loss functions 

        Args: 
            is_training: Are we training?
            batch_size: How many images per batch
            image_h: Image height
            image_w: Image weight
            model: Generator network model - either RESITUAL or MULTISCALE
            texture: Target texture (required if is_training is true)
            inf_noise: Range for input noise when performming style transfer (only applicable if model == MULTISCALE)
    """
    def __init__(self, is_training, batch_size, image_h, image_w, model=RESIDUAL, texture=None, inf_noise=1.):
        self.batch_size = batch_size
        self.image = tf.placeholder(tf.float32, shape=(batch_size, image_h, image_w, 3), name="image_in")
        self.model_name=model
        if model == MULTISCALE:
            K = 6          
            self.generator = GeneratorNetwork(self.image/255.0, is_training, 6, inf_noise)
        elif model == RESIDUAL:
            self.generator = ResidualGenerator(self.image/255.0, is_training)
        self.is_training = is_training
        if is_training:
            assert texture is not None
            with tf.name_scope('VGG'):
                self.descriptor_loss = DescriptorLoss(self.generator.out, texture, batch_size, content=self.image)

    def build_loss(self, session, texture_weight=15, tv=500):
        if self.is_training:
            with tf.name_scope('loss'):     
                self.loss = self.descriptor_loss.build(session, self.generator.image_in, texture_weight)
                if tv > 0:
                    print("tv loss %d" % tv)
                    with tf.name_scope('tv_loss'):
                        batches, h, w, c = self.generator.out.get_shape().as_list()
                        x = self.generator.out[:,1:,:,:]
                        x_1 = self.generator.out[:,:(h-1),:,:]
                        y = self.generator.out[:,:,1:,:]
                        y_1 = self.generator.out[:,:,:w-1,:]
                        x_var = tf.nn.l2_loss(x - x_1)
                        y_var = tf.nn.l2_loss(y - y_1)
                        x_n = batches * (h-1) * w * c
                        y_n = batches * h * (w-1) * c
                        tv_loss = tv * (x_var/x_n + y_var/y_n)
                    self.loss = self.loss + tv_loss

            loss_summary_name = "loss"
            self.summary = tf.scalar_summary(loss_summary_name, self.loss)
            image_summary_name = "out"
            self.image_summary = tf.image_summary(image_summary_name, self.generator.out + utils.MEAN_VALUES, max_images=3)
            input_summary_name = "in"
            self.input_summary = tf.image_summary(input_summary_name, self.image + utils.MEAN_VALUES, max_images=3)

            self.merged = tf.merge_all_summaries()

            self.global_step = tf.Variable(0, name='global_step', trainable=False)

            return self.loss

    def add_noise_to_feed(self, feed):
        for tensor in self.generator.noise_inputs:
            _, h, w, c = tensor.get_shape().as_list()
            feed[tensor] = np.random.uniform(0., 1., (self.batch_size, h, w, c)).astype('float32')     
        return feed

    def run_epoch(self, session, train_op, train_writer, batch_gen=None, num_iterations=NUM_ITERATIONS, output_dir="output", write_image=False):
        epoch_size = num_iterations
        start_time = time.time()
        image_skip = 1 if epoch_size < 5 else epoch_size / 5
        summary_skip = 1 if epoch_size < 25 else epoch_size / 25
        for step in range(epoch_size):
            if self.model_name == MULTISCALE:
                feed = self.add_noise_to_feed({})
            else:
                feed = {}
            batch = batch_gen.get_batch()
            feed[self.image] = batch
            if self.is_training:
                ops = [train_op, self.loss, self.merged, self.image_summary, self.input_summary, self.generator.out, self.global_step]
                _, loss, summary, image_summary, input_summary, last_out, global_step = session.run(ops, feed_dict=feed)
                if write_image and step % image_skip == 0:
                    utils.write_image(os.path.join('%s/images/valid_%d.png' % (output_dir, step)), last_out)
                if train_writer != None:
                    if step % summary_skip == 0:
                        train_writer.add_summary(summary, global_step)
                        train_writer.flush()
                    if step % image_skip == 0:
                        train_writer.add_summary(input_summary)
                        train_writer.flush()
                        train_writer.add_summary(image_summary)
                        train_writer.flush()
            else:
                ops = self.generator.out
                last_out = session.run(ops, feed_dict=feed)
                loss = summary = image_summary = input_summary = global_step = None
        return loss, summary, image_summary, last_out, global_step
