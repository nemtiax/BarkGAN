import tensorflow as tf
from ops import *

class Generator(object):
    def __init__(self,batch_size=16):
        self.batch_size = batch_size
        self.has_built=False

    def build(self,z):
        with tf.variable_scope("Generator",reuse=self.has_built):
            # upsample_z = linear(z,2*2*512,'g_upsample_lin')
            # reshape_z = tf.reshape(upsample_z,[self.batch_size,2,2,512])
            #
            # h0,_,_ = deconv2d(reshape_z,[self.batch_size,4,4,128],name='g_h0_deconv', with_w=True)
            # h0 = lrelu(batch_norm(h0,scope='g_h0_deconv'))
            # h1,_,_ = deconv2d(h0,[self.batch_size,8,8,32],name='g_h1_deconv', with_w=True)
            # h1 = lrelu(batch_norm(h1,scope='g_h1_deconv'))
            # h2,_,_ = deconv2d(h1,[self.batch_size,16,16,16],name='g_h2_deconv', with_w=True)
            # h2 = lrelu(batch_norm(h2,scope='g_h2_deconv'))
            # h3,_,_ = deconv2d(h2,[self.batch_size,32,32,8],name='g_h3_deconv', with_w=True)
            # h3 = lrelu(batch_norm(h3,scope='g_h3_deconv'))
            #
            # h4,_,_ = deconv2d(h3,[self.batch_size,64,64,3],name='g_h4_deconv', with_w=True)
            # h4 = lrelu(batch_norm(h4,scope='g_h4_deconv'))
            #
            # output = tf.nn.tanh(h4)
            #
            # self.has_built=True
            #
            # return output
            upsample_z = linear(z,4*4*1024,'g_upsample_lin')
            reshape_z = tf.reshape(upsample_z,[self.batch_size,4,4,1024])

            h0 = deconv2d(reshape_z,[self.batch_size,8,8,512],name='g_h0_deconv')
            self.h0 = lrelu(batch_norm(h0,scope='g_h0_deconv'))
            h1 = deconv2d(self.h0,[self.batch_size,16,16,256],name='g_h1_deconv')
            h1 = lrelu(batch_norm(h1,scope='g_h1_deconv'))
            h2 = deconv2d(h1,[self.batch_size,32,32,128],name='g_h2_deconv')
            h2 = lrelu(batch_norm(h2,scope='g_h2_deconv'))


            h3 = deconv2d(h2,[self.batch_size,64,64,3],name='g_h3_deconv')
            h3 = lrelu(batch_norm(h3,scope='g_h3_deconv'))



            h4 = deconv2d(h3,[self.batch_size,64,64,3],d_h=1, d_w=1,name='g_h4_deconv')
            h4 = lrelu(batch_norm(h4,scope='g_h4_deconv'))

            output = tf.nn.tanh(h4)


            self.has_built=True

        with tf.variable_scope("Generator",reuse=True):
            h2_zoom = tf.image.resize_bicubic(h2,[64,64])
            h3_zoomed = deconv2d(h2_zoom,[self.batch_size,128,128,3],name='g_h3_deconv')
            h3_zoomed = lrelu(batch_norm(h3_zoomed,scope='g_h3_deconv',is_training=False))
            h4_zoomed = deconv2d(h3_zoomed,[self.batch_size,128,128,3],d_h=1, d_w=1,name='g_h4_deconv')
            h4_zoomed = lrelu(batch_norm(h4_zoomed,scope='g_h4_deconv',is_training=False))
            self.output_zoomed = tf.nn.tanh(h4_zoomed)
        return output
