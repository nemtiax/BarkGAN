import tensorflow as tf
class Pair(object):
    def __init__(self,generator,discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = generator.batch_size

    def build(self):
        self.real_images = tf.placeholder(tf.float32,[self.batch_size,64,64,3],name='real_images')
        self.fake_images = tf.placeholder(tf.float32,[self.batch_size,64,64,3],name='fake_images')
        self.z = tf.placeholder(tf.float32,[self.batch_size,100],name='z')
        self.G = self.generator.build(self.z)

        self.D_real_fake,self.D_real_fake_logits = self.discriminator.build(self.real_images,self.G)
        self.D_fake_real,self.D_fake_real_logits = self.discriminator.build(self.G,self.real_images)

        self.D_fake_comp,self.D_fake_comp_logits = self.discriminator.build(self.G,self.fake_images)
        self.D_comp_fake,self.D_comp_fake_logits = self.discriminator.build(self.fake_images,self.G)


    def build_loss(self):
        self.d_loss_real_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_real_fake_logits, tf.zeros_like(self.D_real_logits)))
        self.d_loss_fake_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_real_logits,tf.ones_like(self.D_fake_logits)))

        self.g_loss_fake_comp = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_comp,tf.zeros_like(self.D_fake_comp)))
        self.g_loss_comp_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_comp_fake,tf.ones_like(self.D_comp_fake)))
        
        self.d_loss = self.d_loss_real+self.d_loss_fake

    def build_train_ops(self):
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'Generator' in var.name]

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
