# -*- coding: utf-8 -*-

import tensorflow as tf

x = tf.placeholder("double") 
y = tf.placeholder("double")
z = tf.mul(x, y)

with tf.Session() as sess:
    tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'basic.pb', as_text=False)