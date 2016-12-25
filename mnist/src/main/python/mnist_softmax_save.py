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
import tensorflow as tf

def main(_):
  # Create the model
  x = tf.placeholder(tf.float32, [None, 784], name = "x")
  W = tf.Variable(tf.zeros([784, 10]), name = "weights")
  b = tf.Variable(tf.zeros([10]), name = "bias")
  y = tf.add(tf.matmul(x, W), b, name = "y")

  # Define loss and optimizer
  y_ = tf.placeholder(tf.float32, [None, 10], name = "y_")


  init_op = tf.initialize_all_variables()
  saver = tf.train.Saver()
  saver_def = saver.as_saver_def()
  
  # print("Init all name: %s" % init_op.name)  
  # print("saver def: %s" % saver_def.filename_tensor_name)
  # print("restore op name: %s" % saver_def.restore_op_name)
  
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(y, y_))
  train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy, name = "train_step")


  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1), name = "corrcect_prediction")
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name = "accuracy")
  
  sess = tf.Session()
  tf.train.write_graph(sess.graph_def, '/tmp/my-model', 'softmax.pb', as_text=False)
  
if __name__ == '__main__':
  tf.app.run()
