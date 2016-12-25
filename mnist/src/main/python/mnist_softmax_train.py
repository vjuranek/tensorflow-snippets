# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A very simple MNIST classifier.

See extensive documentation at
http://tensorflow.org/tutorials/mnist/beginners/index.md
"""
import argparse

# Import data
from tensorflow.examples.tutorials.mnist import input_data

import tensorflow as tf

FLAGS = None


def main(_):
  mnist = input_data.read_data_sets(FLAGS.data_dir, one_hot=True)

  # Create the model
  x = tf.placeholder(tf.float32, [None, 784], name = "x")
  W = tf.Variable(tf.zeros([784, 10]), name = "weights")
  b = tf.Variable(tf.zeros([10]), name = "bias")
  y = tf.add(tf.matmul(x, W), b, name = "y")

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10], name = "y_")

  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


  saver = tf.train.Saver()
  with tf.Session() as sess:
    # Train
    tf.initialize_all_variables().run()
    for _ in range(1000):
      batch_xs, batch_ys = mnist.train.next_batch(100)
      sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
      
    save_path = saver.save(sess, "/tmp/my-model/softmax.ckpt")
    print("Model saved in file: %s" % save_path)
  
if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_dir', type=str, default='/tmp/mnist/data',
                      help='Directory for storing data')
  FLAGS = parser.parse_args()
  tf.app.run()
