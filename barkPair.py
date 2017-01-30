class Pair(object):
    def __init__(self,generator,discriminator):
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = generator.batch_size

    def build(self):
        self.real_images = tf.placeholder(tf.float32,[self.batch_size,32,32,3],name='real_images')
        self.z = tf.placeholder(tf.float32,[self.batch_size,100],name='z')
        self.G = self.generator.build(self.z)
        self.D_real,self.D_real_logits = self.discriminator.build(self.real_images)
        self.D_fake,self.D_fake_logits = self.discriminator.build(self.G)

    def build_loss(self):
        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_real_logits, tf.ones_like(self.D_real_logits)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_logits,tf.zeros_like(self.D_fake_logits)))

        self.g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_fake_logits,tf.ones_like(self.D_fake_logits)))
        self.d_loss = self.d_loss_real+self.d_loss_fake

    def build_train_ops(self):
        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'Discriminator' in var.name]
        self.g_vars = [var for var in t_vars if 'Generator' in var.name]

        self.d_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(0.0002, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
