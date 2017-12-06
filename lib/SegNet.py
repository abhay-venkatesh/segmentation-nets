# Import modules for building the network
import tensorflow as tf
import numpy as np
import scipy.io
from math import ceil
import cv2
from utils.DatasetReader import DatasetReader
from PIL import Image

class SegNet:
  ''' Network described by,
  https://arxiv.org/pdf/1505.04366v1.pdf
  and https://arxiv.org/pdf/1505.07293.pdf
  and https://arxiv.org/pdf/1511.00561.pdf '''
  self.checkpoint_directory = '../checkpoints/'

  def __init__(self):
    # Build the network
    self.build()

    # Begin a TensorFlow session
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

    # Make saving trained weights and biases possible
    self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours = 1)

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def pool_layer(self, x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  def unravel_argmax(self, argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.stack(output_list)

  # Implementation idea from: https://github.com/tensorflow/tensorflow/issues/2169
  def unpool_layer2x2(self, x, raveled_argmax, out_shape):
    argmax = self.unravel_argmax(raveled_argmax, tf.to_int64(out_shape))
    output = tf.zeros([out_shape[1], out_shape[2], out_shape[3]])
    height = tf.shape(output)[0]
    width = tf.shape(output)[1]
    channels = tf.shape(output)[2]
    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [((width + 1) // 2) * ((height + 1) // 2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, (height + 1) // 2, (width + 1) // 2, 1])
    t2 = tf.squeeze(argmax)
    t2 = tf.stack((t2[0], t2[1]), axis=0)
    t2 = tf.transpose(t2, perm=[3, 1, 2, 0])
    t = tf.concat([t2, t1], 3)
    indices = tf.reshape(t, [((height + 1) // 2) * ((width + 1) // 2) * channels, 3])
    x1 = tf.squeeze(x)
    x1 = tf.reshape(x1, [-1, channels])
    x1 = tf.transpose(x1, perm=[1, 0])
    values = tf.reshape(x1, [-1])
    delta = tf.SparseTensor(indices, values, tf.to_int64(tf.shape(output)))
    return tf.expand_dims(tf.sparse_tensor_to_dense(tf.sparse_reorder(delta)), 0)
        

  # TODO: Add batch normalization
  def conv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
    W = self.weight_variable(W_shape)
    b = self.bias_variable([b_shape])
    output = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding=padding) + b
    return tf.nn.relu(output)

  def deconv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
    W = self.weight_variable(W_shape)
    b = self.bias_variable([b_shape])
    x_shape = tf.shape(x)
    out_shape = tf.stack([x_shape[0], x_shape[1], x_shape[2], W_shape[2]])
    return tf.nn.conv2d_transpose(x, W, out_shape, [1, 1, 1, 1], padding=padding) + b

  def build(self):
    # Declare placeholders
    self.x = tf.placeholder(tf.float32, shape=(1, None, None, 3))
    self.y = tf.placeholder(tf.int64, shape=(1, None, None))
    expected = tf.expand_dims(self.y, -1)
    self.rate = tf.placeholder(tf.float32, shape=[])

    # First encoder
    conv_1_1 = self.conv_layer(self.x, [3, 3, 3, 64], 64, 'conv_1_1')
    conv_1_2 = self.conv_layer(conv_1_1, [3, 3, 64, 64], 64, 'conv_1_2')
    pool_1, pool_1_argmax = self.pool_layer(conv_1_2)

    # Second encoder
    conv_2_1 = self.conv_layer(pool_1, [3, 3, 64, 128], 128, 'conv_2_1')
    conv_2_2 = self.conv_layer(conv_2_1, [3, 3, 128, 128], 128, 'conv_2_2')
    pool_2, pool_2_argmax = self.pool_layer(conv_2_2)

    # Third encoder
    conv_3_1 = self.conv_layer(pool_2, [3, 3, 128, 256], 256, 'conv_3_1')
    conv_3_2 = self.conv_layer(conv_3_1, [3, 3, 256, 256], 256, 'conv_3_2')
    conv_3_3 = self.conv_layer(conv_3_2, [3, 3, 256, 256], 256, 'conv_3_3')
    pool_3, pool_3_argmax = self.pool_layer(conv_3_3)

    # Fourth encoder
    conv_4_1 = self.conv_layer(pool_3, [3, 3, 256, 512], 512, 'conv_4_1')
    conv_4_2 = self.conv_layer(conv_4_1, [3, 3, 512, 512], 512, 'conv_4_2')
    conv_4_3 = self.conv_layer(conv_4_2, [3, 3, 512, 512], 512, 'conv_4_3')
    pool_4, pool_4_argmax = self.pool_layer(conv_4_3)

    # Fifth encoder
    conv_5_1 = self.conv_layer(pool_4, [3, 3, 512, 512], 512, 'conv_5_1')
    conv_5_2 = self.conv_layer(conv_5_1, [3, 3, 512, 512], 512, 'conv_5_2')
    conv_5_3 = self.conv_layer(conv_5_2, [3, 3, 512, 512], 512, 'conv_5_3')
    pool_5, pool_5_argmax = self.pool_layer(conv_5_3)

    # Fully connected layers between the encoder and decoder
    fc_6 = self.conv_layer(pool_5, [7, 7, 512, 4096], 4096, 'fc_6')
    fc_7 = self.conv_layer(fc_6, [1, 1, 4096, 4096], 4096, 'fc_7')

    # Single deconv before beginning decoding
    deconv_fc_6 = self.deconv_layer(fc_7, [7, 7, 512, 4096], 512, 'fc6_deconv')

    # First decoder
    unpool_5 = self.unpool_layer2x2(deconv_fc_6, pool_5_argmax, tf.shape(conv_5_3))
    deconv_5_3 = self.deconv_layer(unpool_5, [3, 3, 512, 512], 512, 'deconv_5_3')
    deconv_5_2 = self.deconv_layer(deconv_5_3, [3, 3, 512, 512], 512, 'deconv_5_2')
    deconv_5_1 = self.deconv_layer(deconv_5_2, [3, 3, 512, 512], 512, 'deconv_5_1')

    # Second decoder
    unpool_4 = self.unpool_layer2x2(deconv_5_1, pool_4_argmax, tf.shape(conv_4_3))
    deconv_4_3 = self.deconv_layer(unpool_4, [3, 3, 512, 512], 512, 'deconv_4_3')
    deconv_4_2 = self.deconv_layer(deconv_4_3, [3, 3, 512, 512], 512, 'deconv_4_2')
    deconv_4_1 = self.deconv_layer(deconv_4_2, [3, 3, 256, 512], 256, 'deconv_4_1')

    # Third decoder
    unpool_3 = self.unpool_layer2x2(deconv_4_1, pool_3_argmax, tf.shape(conv_3_3))
    deconv_3_3 = self.deconv_layer(unpool_3, [3, 3, 256, 256], 256, 'deconv_3_3')
    deconv_3_2 = self.deconv_layer(deconv_3_3, [3, 3, 256, 256], 256, 'deconv_3_2')
    deconv_3_1 = self.deconv_layer(deconv_3_2, [3, 3, 128, 256], 128, 'deconv_3_1')

    # Fourth decoder
    unpool_2 = self.unpool_layer2x2(deconv_3_1, pool_2_argmax, tf.shape(conv_2_2))
    deconv_2_2 = self.deconv_layer(unpool_2, [3, 3, 128, 128], 128, 'deconv_2_2')
    deconv_2_1 = self.deconv_layer(deconv_2_2, [3, 3, 64, 128], 64, 'deconv_2_1')

    # Fifth decoder
    unpool_1 = self.unpool_layer2x2(deconv_2_1, pool_1_argmax, tf.shape(conv_1_2))
    deconv_1_2 = self.deconv_layer(unpool_1, [3, 3, 64, 64], 64, 'deconv_1_2')
    deconv_1_1 = self.deconv_layer(deconv_1_2, [3, 3, 32, 64], 32, 'deconv_1_1')

    # Produce class scores
    preds = self.deconv_layer(deconv_1_1, [1, 1, 27, 32], 27, 'preds')
    self.logits = tf.reshape(preds, (-1, 27))

    # Prepare network for training
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(expected, [-1]), logits=logits, name='x_entropy')
    self.loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
    self.train_step = tf.train.AdamOptimizer(self.rate).minimize(self.loss)

    # Metrics
    self.accuracy = tf.contrib.metrics.accuracy(preds, self.y, name='accuracy')

  def train(self, num_iterations=10000, learning_rate=1e-6):
    dataset = DatasetReader()
    
    # Begin Training
    for i in range(num_iterations):

      # One training step
      image, ground_truth = dataset.next_train_pair()
      feed_dict = {self.x: [image], self.y: [ground_truth], self.rate: learning_rate}
      print('run train step: '+str(i))
      self.train_step.run(session=self.session, feed_dict=feed_dict)

      # Print loss every 10 iterations
      if i % 10 == 0:
        train_loss = self.session.run(self.loss, feed_dict=feed_dict)
        print("Step: %d, Train_loss:%g" % (i, train_loss))

      # Run against validation dataset for 100 iterations
      if i % 100 == 0:
        image, ground_truth = dataset.next_val_pair()
        feed_dict = {self.x: [image], self.y: [ground_truth], self.rate: learning_rate}
        val_loss = self.session.run(self.loss, feed_dict=feed_dict)
        val_accuracy = self.session.run(self.accuracy, feed_dict=feed_dict)
        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), valid_loss))
        print("%s ---> Validation_accuracy: %g" % (datetime.datetime.now(), valid_accuracy))

        # Save the model variables
        self.saver.save(self.session, self.checkpoint_directory + 'segnet', global_step = i)

  def test(self, learning_rate=1e-6):
    dataset = DatasetReader()
    image, ground_truth = dataset.next_test_pair() 
    feed_dict = {self.x: [image], self.y: [ground_truth], self.rate: learning_rate}
    prediction = self.session(self.logits, feed_dict=feed_dict)
    img = Image.fromarray(prediction, 'L')
    img.save('prediction.png')
