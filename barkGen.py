class Generator(object):
    def __init__(self,batch_size=16):
        self.batch_size = batch_size
        self.has_built=False

    def build(self,z):
        with tf.variable_scope("Generator",reuse=self.has_built):
            upsample_z = linear(z,2*2*512,'g_upsample_lin')

            h0,_,_ = deconv2d(upsample_z,[self.batch_size,4,4,128],name='g_h0_deconv', with_w=True)
            h0 = lrelu(batch_norm(h0,scope='g_h0_deconv'))
            h1,_,_ = deconv2d(h0,[self.batch_size,8,8,32],name='g_h1_deconv', with_w=True)
            h1 = lrelu(batch_norm(h1,scope='g_h1_deconv'))
            h2,_,_ = deconv2d(h1,[self.batch_size,16,16,8],name='g_h2_deconv', with_w=True)
            h2 = lrelu(batch_norm(h2,scope='g_h2_deconv'))
            h3,_,_ = deconv2d(h2,[self.batch_size,32,32,3],name='g_h3_deconv', with_w=True)
            h3 = lrelu(batch_norm(h3,scope='g_h3_deconv'))

            output = tf.nn.tanh(h3)

            self.has_built=True

            return output
