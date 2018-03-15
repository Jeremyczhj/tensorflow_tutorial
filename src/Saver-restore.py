import tensorflow as tf
import numpy as np

W = tf.Variable(np.arange(6).reshape((2,3)),dtype=tf.float32,name='weights')
b = tf.Variable(np.arange(3).reshape((1,3)),dtype=tf.float32,name='biases')

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess,"E://log/save_net.ckpt")
    print("weights:",sess.run(W))
    print("biases:",sess.run(b))
    #tensorflow目前只能保存variable，不能保存整个网络框架，需要我们自己定义
