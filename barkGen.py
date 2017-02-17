import tensorflow as tf
from ops import *

class Generator(object):
    def __init__(self,batch_size=16):
        self.batch_size = batch_size
        self.has_built=False

    def build(self,z):
        with tf.variable_scope("Generator",reuse=self.has_built):

            upsample_z = linear(z,4*4*1024,'g_upsample_lin')
            reshape_z = tf.reshape(upsample_z,[self.batch_size,4,4,1024])

            h0 = deconv2d(reshape_z,[self.batch_size,8,8,512],name='g_h0_deconv')
            h0 = lrelu(batch_norm(h0,scope='g_h0_deconv'))
            h1 = deconv2d(h0,[self.batch_size,16,16,256],name='g_h1_deconv')
            h1 = lrelu(batch_norm(h1,scope='g_h1_deconv'))
            h2 = deconv2d(h1,[self.batch_size,32,32,128],name='g_h2_deconv')
            h2 = lrelu(batch_norm(h2,scope='g_h2_deconv'))
            h3 = deconv2d(h2,[self.batch_size,64,64,3],name='g_h3_deconv')
            h3 = lrelu(batch_norm(h3,scope='g_h3_deconv'))
            h4 = deconv2d(h3,[self.batch_size,64,64,3],d_h=1, d_w=1,name='g_h4_deconv')
            h4 = lrelu(batch_norm(h4,scope='g_h4_deconv'))

            output = tf.nn.tanh(h4)

            self.has_built=True

            return output
