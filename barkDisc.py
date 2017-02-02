import tensorflow as tf
from ops import *

class Discriminator(object):
    def __init__(self,batch_size=16):
        self.batch_size = batch_size
        self.has_built=False

    def build(self,input_images):
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



            h0 = lrelu(conv2d(input_images,64,name='d_h0_conv'))
            h1 = lrelu(batch_norm(conv2d(h0,128,name='d_h1_conv'),scope='d_h1_conv'))
            h2 = lrelu(batch_norm(conv2d(h1,256,name='d_h2_conv'),scope='d_h2_conv'))
            h3 = lrelu(batch_norm(conv2d(h2,512,name='d_h3_conv'),scope='d_h3_conv'))
            h4 = linear(tf.reshape(h3,[self.batch_size,-1]),1,'d_h3_lin')
            self.has_built=True
            return tf.nn.sigmoid(h4),h4
