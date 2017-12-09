import tensorflow as tf
import numpy as np
import scipy.io
from math import ceil
import cv2
from utils.DatasetReader import DatasetReader
from PIL import Image
import datetime
import os

class FCN:

  def load_vgg_weights(self):
    """ Use the VGG model trained on
      imagent dataset as a starting point for training """

    # Download model if not existing
    # TODO: wget does not work for windows
    # TODO: Doesn't work on linux either
    try:
      vgg_path = "models/imagenet-vgg-verydeep-19.mat"
      vgg_mat = scipy.io.loadmat(vgg_path)
    except (OSError, IOError) as e:
      import wget
      vgg_url = "http://www.vlfeat.org/matconvnet/models/beta16/imagenet-vgg-verydeep-19.mat"
      wget.download(vgg_url, out='./models/imagenet-vgg-verydeep-19.mat')

    self.vgg_params = np.squeeze(vgg_mat['layers'])
    self.layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4')

  def __init__(self):
    # Load VGG model weights to initialize network weights to
    self.load_vgg_weights()

    # Build the network
    self.build()

    # Begin a TensorFlow session
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

    # Make saving trained weights and biases possible
    self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours = 1)
    self.checkpoint_directory = './checkpoints/'

  def vgg_weight_and_bias(self, name, W_shape, b_shape):
    """ 
      Initializes weights and biases to the pre-trained VGG model.
      
      Args:
        name: name of the layer for which you want to initialize weights
        W_shape: shape of weights tensor expected
        b_shape: shape of bias tensor expected
      returns:
        w_var: Initialized weight variable
        b_var: Initialized bias variable
    """
    if name not in self.layers:
      raise KeyError("Layer missing in VGG model or mispelled. ")
    else:
      w, b = self.vgg_params[self.layers.index(name)][0][0][0][0]
      init_w = tf.constant(value=np.transpose(w, (1, 0, 2, 3)), dtype=tf.float32, shape=W_shape)
      init_b = tf.constant(value=b.reshape(-1), dtype=tf.float32, shape=b_shape)
      w_var = tf.Variable(init_w)
      b_var = tf.Variable(init_b)
      return w_var, b_var 

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

  def pool_layer(self, x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
  def conv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
    # Pass b_shape as list because need the object to be iterable for the constant initializer
    W, b = self.vgg_weight_and_bias(name, W_shape, [b_shape])

    output = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding=padding) + b
    return tf.nn.relu(output)

  def upscoreLayer(bottom, shape,num_classes, name,ksize=4, stride=2):
    strides = [1, stride, stride, 1]
    in_features = bottom.get_shape()[3].value

    if shape is None:
        # Compute shape out of Bottom
        in_shape = tf.shape(bottom)
        h = ((in_shape[1] - 1) * stride) + 1
        w = ((in_shape[2] - 1) * stride) + 1
        new_shape = [in_shape[0], h, w, num_classes]
    else:
        new_shape = [shape[0], shape[1], shape[2], num_classes]
    output_shape = tf.stack(new_shape) 

    f_shape = [ksize, ksize, num_classes, in_features]
    width = f_shape[0]
    heigh = f_shape[0]
    f = ceil(width/2.0)
    c = (2 * f - 1 - f % 2) / (2.0 * f)
    bilinear = np.zeros([f_shape[0], f_shape[1]])
    for x in range(width):
        for y in range(heigh):
            value = (1 - abs(x / f - c)) * (1 - abs(y / f - c))
            bilinear[x, y] = value
    weights = np.zeros(f_shape)
    for i in range(f_shape[2]):
        weights[:, :, i, i] = bilinear
    init = tf.constant_initializer(value=weights,dtype=tf.float32)
    weights = tf.get_variable(initializer=init,shape=weights.shape,dtype=tf.float32,name="%s_w"%name)
    deconv = tf.nn.conv2d_transpose(bottom, weights, output_shape,strides=strides, padding='SAME', name = name)
    init = tf.constant_initializer(value=0.0,dtype=tf.float32)
    bias = tf.get_variable(name="%s_b"%name,initializer=init,shape=[num_classes],dtype=tf.float32)
    return tf.nn.bias_add(deconv, bias)

  def build(self):
    # Declare placeholders
    self.x = tf.placeholder(tf.float32, shape=(1, None, None, 3))
    self.y = tf.placeholder(tf.int64, shape=(1, None, None))
    expected = tf.expand_dims(self.y, -1)
    self.rate = tf.placeholder(tf.float32, shape=[])

    # First encoder
    conv_1_1 = self.conv_layer(self.x, [3, 3, 3, 64], 64, 'conv1_1')
    conv_1_2 = self.conv_layer(conv_1_1, [3, 3, 64, 64], 64, 'conv1_2')
    pool_1 = self.pool_layer(conv_1_2)

    # Second encoder
    conv_2_1 = self.conv_layer(pool_1, [3, 3, 64, 128], 128, 'conv2_1')
    conv_2_2 = self.conv_layer(conv_2_1, [3, 3, 128, 128], 128, 'conv2_2')
    pool_2  = self.pool_layer(conv_2_2)

    # Third encoder
    conv_3_1 = self.conv_layer(pool_2, [3, 3, 128, 256], 256, 'conv3_1')
    conv_3_2 = self.conv_layer(conv_3_1, [3, 3, 256, 256], 256, 'conv3_2')
    conv_3_3 = self.conv_layer(conv_3_2, [3, 3, 256, 256], 256, 'conv3_3')
    pool_3 = self.pool_layer(conv_3_3)

    # Fourth encoder
    conv_4_1 = self.conv_layer(pool_3, [3, 3, 256, 512], 512, 'conv4_1')
    conv_4_2 = self.conv_layer(conv_4_1, [3, 3, 512, 512], 512, 'conv4_2')
    conv_4_3 = self.conv_layer(conv_4_2, [3, 3, 512, 512], 512, 'conv4_3')
    pool_4 = self.pool_layer(conv_4_3)

    # Fifth encoder
    conv_5_1 = self.conv_layer(pool_4, [3, 3, 512, 512], 512, 'conv5_1')
    conv_5_2 = self.conv_layer(conv_5_1, [3, 3, 512, 512], 512, 'conv5_2')
    conv_5_3 = self.conv_layer(conv_5_2, [3, 3, 512, 512], 512, 'conv5_3')
    pool_5 = self.pool_layer(conv_5_3)

    # Produce class scores
    preds = self.deconv_layer(deconv_1_1, [1, 1, 28, 32], 28, 'preds')
    self.logits = tf.reshape(preds, (-1, 28))

    # Prepare network for training
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(expected, [-1]), logits=self.logits, name='x_entropy')
    self.loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
    self.train_step = tf.train.AdamOptimizer(self.rate).minimize(self.loss)

    # Metrics
    predicted_image = tf.argmax(preds, axis=3)
    self.accuracy = tf.contrib.metrics.accuracy(tf.cast(predicted_image, tf.int64), self.y, name='accuracy')

  def restore_session(self):
    global_step = 0

    if not os.path.exists(self.checkpoint_directory):
      raise IOError(self.checkpoint_directory + ' does not exist.')
    else:
      path = tf.train.get_checkpoint_state(self.checkpoint_directory)
      if path is None:
        pass
      else:
        self.saver.restore(self.session, path.model_checkpoint_path)
        global_step = int(path.model_checkpoint_path.split('-')[-1])

    return global_step

  
  def train(self, num_iterations=10000, learning_rate=1e-6):
    # Restore previous session if exists
    current_step = self.restore_session()

    dataset = DatasetReader()
    
    # Begin Training
    for i in range(current_step, num_iterations):

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
        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), val_loss))
        print("%s ---> Validation_accuracy: %g" % (datetime.datetime.now(), val_accuracy))

        # Save the model variables
        self.saver.save(self.session, self.checkpoint_directory + 'segnet', global_step = i)

  def test(self, learning_rate=1e-6):
    dataset = DatasetReader()
    image, ground_truth = dataset.next_test_pair() 
    feed_dict = {self.x: [image], self.y: [ground_truth], self.rate: learning_rate}
    prediction = self.session(self.logits, feed_dict=feed_dict)
    img = Image.fromarray(prediction, 'L')
    img.save('prediction.png')
