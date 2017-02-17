import tensorflow as tf
from ops import *

class Discriminator(object):
    def __init__(self,batch_size=16):
        self.batch_size = batch_size
        self.has_built=False

    def build(self,input_images_a,input_images_b):
        with tf.variable_scope("Discriminator",reuse=self.has_built):
            # h0 = lrelu(batch_norm(conv2d(input_images, 8,name='d_h0_conv'),scope='d_h0_conv')) #32x32x8
            # h1 = lrelu(batch_norm(conv2d(h0, 16,name='d_h1_conv'),scope='d_h1_conv')) #16x16x16
            # h2 = lrelu(batch_norm(conv2d(h1, 64,name='d_h2_conv'),scope='d_h2_conv')) #8x8x64
            # h3 = lrelu(batch_norm(conv2d(h2, 128,name='d_h3_conv'),scope='d_h3_conv')) #4x4x128
            # h4 = lrelu(batch_norm(conv2d(h3, 512,name='d_h4_conv'),scope='d_h4_conv')) #2x2x512
            #
            # h5 = linear(tf.reshape(h4, [self.batch_size, -1]), 1, 'd_h5_lin')
            # self.has_built = True
            # return tf.nn.sigmoid(h5),h5



            h0_a = lrelu(conv2d(input_images_a,64,name='d_h0_a_conv'))
            h1_a = lrelu(batch_norm(conv2d(h0_a,128,name='d_h1_a_conv'),scope='d_h1_a_conv'))
            h2_a = lrelu(batch_norm(conv2d(h1_a,256,name='d_h2_a_conv'),scope='d_h2_a_conv'))
            h3_a = lrelu(batch_norm(conv2d(h2_a,512,name='d_h3_a_conv'),scope='d_h3_a_conv'))

            h0_b = lrelu(conv2d(input_images_b,64,name='d_h0_b_conv'))
            h1_b = lrelu(batch_norm(conv2d(h0_b,128,name='d_h1_b_conv'),scope='d_h1_b_conv'))
            h2_b = lrelu(batch_norm(conv2d(h1_b,256,name='d_h2_b_conv'),scope='d_h2_b_conv'))
            h3_b = lrelu(batch_norm(conv2d(h2_b,512,name='d_h3_b_conv'),scope='d_h3_b_conv'))

            h3_combined = tf.concat(3,[h3_a,h3_b])

            h4 = linear(tf.reshape(h3,[self.batch_size,-1]),1,'d_h3_lin')
            self.has_built=True
            return tf.nn.sigmoid(h4),h4
