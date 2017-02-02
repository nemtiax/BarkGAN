import tensorflow as tf
from ops import *

class Generator(object):
    def __init__(self,batch_size=16):
        self.batch_size = batch_size
        self.has_built=False

    def build(self,z,input_images,patch_masks,patch_offsets):
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


            i0 = lrelu(conv2d(input_images,64,name='d_h0_conv'))
            i1 = lrelu(batch_norm(conv2d(i0,128,name='d_h1_conv'),scope='g_i1_conv'))
            i2 = lrelu(batch_norm(conv2d(i1,256,name='d_h2_conv'),scope='g_i2_conv'))
            i3 = lrelu(batch_norm(conv2d(i2,512,name='d_h3_conv'),scope='g_i3_conv'))
            i4 = linear(tf.reshape(i3,[self.batch_size,-1]),100,'g_i3_lin')

            i4 = tf.nn.tanh(i4)

            z = tf.concat(1,[z,i4])


            upsample_z = linear(z,4*4*1024,'g_upsample_lin')
            reshape_z = tf.reshape(upsample_z,[self.batch_size,4,4,1024])

            h0 = deconv2d(reshape_z,[self.batch_size,8,8,512],name='g_h0_deconv')
            h0 = lrelu(batch_norm(h0,scope='g_h0_deconv'))
            h1 = deconv2d(h0,[self.batch_size,16,16,256],name='g_h1_deconv')
            h1 = lrelu(batch_norm(h1,scope='g_h1_deconv'))
            h2 = deconv2d(h1,[self.batch_size,32,32,128],name='g_h2_deconv')
            h2 = lrelu(batch_norm(h2,scope='g_h2_deconv'))
            h3 = deconv2d(h2,[self.batch_size,32,32,3],d_h=1, d_w=1,name='g_h3_deconv')
            h3 = lrelu(batch_norm(h3,scope='g_h3_deconv'))

            output = tf.nn.tanh(h3)

            self.has_built=True


            images_with_hole = input_images*patch_masks

            #split output into a list of 64 patches
            unstacked_patches = tf.unstack(output)
            unstacked_offsets = tf.unstack(patch_offsets)
            padded_patches = []
            for i in range(self.batch_size):
                patch = unstacked_patches[i]
                xOffset = unstacked_offsets[i][0]
                yOffset = unstacked_offsets[i][1]
                paddings = tf.stack([[xOffset,64-(32+xOffset)],[yOffset,64-(32+yOffset)],[0,0]])
                padded_patch = tf.pad(patch,paddings)
                padded_patches.append(padded_patch)

            patched_images = images_with_hole + tf.mul(tf.constant(0.95),padded_patches)

            return patched_images
