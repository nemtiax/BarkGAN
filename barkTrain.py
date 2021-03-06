from barkDisc import Discriminator
from barkGen import Generator
from barkPair import Pair

import os
from glob import glob
from utils import *
from random import shuffle
import tensorflow as tf

class Trainer(object):
    def __init__(self,pair,sess):
        self.pair = pair
        self.batch_size = pair.batch_size
        self.sess = sess

    def load_data(self):
        self.data_files = glob(os.path.join("./data", "all_tiles", "*.png"))
        self.data = [get_image(data_file, 256, is_crop=True, resize_w=64) for data_file in self.data_files]

    def train(self,epochs):

        sample_z = np.random.uniform(-1, 1, size=(self.batch_size,100))
        count = 0
        tf.initialize_all_variables().run()
        for ep in range(epochs):
           self.shuffle_data()
           for batch_index in range(len(self.data)//self.batch_size):
               real_images,z = self.get_batch(batch_index)
               _, g_loss = sess.run([self.pair.g_optim,self.pair.g_loss],feed_dict={self.pair.z: z})
               _, d_loss = sess.run([self.pair.d_optim,self.pair.d_loss],feed_dict={self.pair.real_images: real_images,self.pair.z: z})
               count = count+1
               if(count%100==0):
                   generated_images = sess.run(self.pair.G,feed_dict={self.pair.z: z})
                   save_images(generated_images,[8,8],'./bark_samples/train_{:02d}_{:04d}.png'.format(ep,batch_index))
                   save_images(real_images,[8,8],'./bark_samples/batch_{:02d}_{:04d}.png'.format(ep,batch_index))
                   print('{:4d} - {:4d}:  D_loss: {:.4f}, G_loss: {:.4f}'.format(ep,batch_index,d_loss,g_loss))

    def shuffle_data(self):
        shuffle(self.data)

    def get_batch(self,index):
        batch = self.data[index*self.batch_size:(index+1)*self.batch_size]
        batch_images = np.array(batch).astype(np.float32)
        z = np.random.uniform(-1, 1, size=(self.batch_size,100))
        return batch_images,z

with tf.Session() as sess:
    g = Generator(batch_size=32)
    d = Discriminator(batch_size=32)
    p = Pair(g,d)
    p.build()
    p.build_loss()
    p.build_train_ops()
    t = Trainer(p,sess)
    t.load_data()
    t.train(100)
