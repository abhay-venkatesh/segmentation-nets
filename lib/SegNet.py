import tensorflow as tf
import numpy as np
import scipy.io
from math import ceil
import cv2
from utils.DatasetReader import DatasetReader
from PIL import Image
import datetime
import os
from utils.Logger import Logger
import math

# Only use a single GPU when not testing
if os.name != 'nt': 
  os.environ["CUDA_VISIBLE_DEVICES"] = '1'

# TODO: Not working but batch version is fine
class SegNet:
  ''' Network described by,
  https://arxiv.org/pdf/1511.00561.pdf '''

  def load_vgg_weights(self):
    """ Use the VGG model trained on
      imagent dataset as a starting point for training """

    # REMINDER: Download model if not existing
    vgg_path = "models/imagenet-vgg-verydeep-19.mat"
    vgg_mat = scipy.io.loadmat(vgg_path)

    self.vgg_params = np.squeeze(vgg_mat['layers'])
    self.layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1',
            'conv2_1', 'relu2_1', 'conv2_2', 'relu2_2', 'pool2',
            'conv3_1', 'relu3_1', 'conv3_2', 'relu3_2', 'conv3_3',
            'relu3_3', 'conv3_4', 'relu3_4', 'pool3',
            'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
            'relu4_3', 'conv4_4', 'relu4_4', 'pool4',
            'conv5_1', 'relu5_1', 'conv5_2', 'relu5_2', 'conv5_3',
            'relu5_3', 'conv5_4', 'relu5_4')

  def __init__(self, dataset_directory, num_classes=11):
    self.dataset_directory = dataset_directory

    self.num_classes = num_classes

    self.load_vgg_weights()

    self.build()

    # Begin a TensorFlow session
    config = tf.ConfigProto(allow_soft_placement=True)
    self.session = tf.Session()
    self.session.run(tf.global_variables_initializer())

    # Make saving trained weights and biases possible
    self.saver = tf.train.Saver(max_to_keep = 5, keep_checkpoint_every_n_hours = 1)
    self.checkpoint_directory = './checkpoints/'
    self.logger = Logger()

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
      return self.weight_variable(W_shape), self.bias_variable(b_shape)
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

  def batch_norm_layer(self, inputT, is_training, scope):
    return tf.cond(is_training,
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=True,
                           center=False, updates_collections=None, scope=scope+"_bn"),
          lambda: tf.contrib.layers.batch_norm(inputT, is_training=False,
                           updates_collections=None, center=False, scope=scope+"_bn", reuse = True))

  def pool_layer(self, x):
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  def unpool(self, pool, ind, ksize=(1, 2, 2, 1), scope='unpool'):
    """ Unpooling layer after max_pool_with_argmax.
      Args:
        pool: max pooled output tensor
        ind: argmax indices (produced by tf.nn.max_pool_with_argmax)
        ksize: ksize is the same as for the pool
      Return:
        unpooled: unpooling tensor
      Footnote:
        Implementation idea from: https://github.com/tensorflow/tensorflow/issues/2169
    """
    with tf.variable_scope(scope):
      pooled_shape = tf.shape(pool) 
      flatten_ind = tf.reshape(ind, (pooled_shape[0], pooled_shape[1] * pooled_shape[2] * pooled_shape[3]))
      # sparse indices to dense ones_like matrics
      one_hot_ind = tf.one_hot(flatten_ind,  pooled_shape[1] * ksize[1] * pooled_shape[2] * ksize[2] * pooled_shape[3], on_value=1., off_value=0., axis=-1)
      one_hot_ind = tf.reduce_sum(one_hot_ind, axis=1)
      one_like_mask = tf.reshape(one_hot_ind, (pooled_shape[0], pooled_shape[1] * ksize[1], pooled_shape[2] * ksize[2], pooled_shape[3]))
      # resize input array to the output size by nearest neighbor
      img = tf.image.resize_nearest_neighbor(pool, [pooled_shape[1] * ksize[1], pooled_shape[2] * ksize[2]])
      unpooled = tf.multiply(img, tf.cast(one_like_mask, img.dtype))
      return unpooled 

  def unravel_argmax(self, argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.stack(output_list)
  
  def unpool_layer2x2(self, x, raveled_argmax, out_shape):
    ''' Implementation idea from: 
        https://github.com/tensorflow/tensorflow/issues/2169 '''

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
        
  def conv_layer(self, x, W_shape, name, padding='SAME'):
    # Pass b_shape as list because need the object to be iterable for the constant initializer
    out_channel = W_shape[3]
    W, b = self.vgg_weight_and_bias(name, W_shape, [out_channel])

    output = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding=padding) + b
    return tf.nn.relu(output)

  def conv_layer_with_bn(self, x, W_shape, train_phase, name, padding='SAME'):
    out_channel = W_shape[3]
    with tf.variable_scope(name) as scope:
      W, b = self.vgg_weight_and_bias(name, W_shape, [out_channel])
      output = tf.nn.conv2d(x, W, strides=[1,1,1,1], padding=padding) + b
      return tf.nn.relu(self.batch_norm_layer(output, train_phase, scope.name))

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
    self.train_phase = tf.placeholder(tf.bool, name='train_phase')

    # First encoder
    conv_1_1 = self.conv_layer_with_bn(self.x, [3, 3, 3, 64], self.train_phase, 'conv1_1')
    conv_1_2 = self.conv_layer_with_bn(conv_1_1, [3, 3, 64, 64], self.train_phase, 'conv1_2')
    pool_1, pool_1_argmax = self.pool_layer(conv_1_2)

    # Second encoder
    conv_2_1 = self.conv_layer_with_bn(pool_1, [3, 3, 64, 128], self.train_phase, 'conv2_1')
    conv_2_2 = self.conv_layer_with_bn(conv_2_1, [3, 3, 128, 128], self.train_phase, 'conv2_2')
    pool_2, pool_2_argmax = self.pool_layer(conv_2_2)

    # Third encoder
    conv_3_1 = self.conv_layer_with_bn(pool_2, [3, 3, 128, 256], self.train_phase, 'conv3_1')
    conv_3_2 = self.conv_layer_with_bn(conv_3_1, [3, 3, 256, 256], self.train_phase, 'conv3_2')
    conv_3_3 = self.conv_layer_with_bn(conv_3_2, [3, 3, 256, 256], self.train_phase, 'conv3_3')
    pool_3, pool_3_argmax = self.pool_layer(conv_3_3)

    # Fourth encoder
    conv_4_1 = self.conv_layer_with_bn(pool_3, [3, 3, 256, 512], self.train_phase, 'conv4_1')
    conv_4_2 = self.conv_layer_with_bn(conv_4_1, [3, 3, 512, 512], self.train_phase, 'conv4_2')
    conv_4_3 = self.conv_layer_with_bn(conv_4_2, [3, 3, 512, 512], self.train_phase, 'conv4_3')
    pool_4, pool_4_argmax = self.pool_layer(conv_4_3)

    # Fifth encoder
    conv_5_1 = self.conv_layer_with_bn(pool_4, [3, 3, 512, 512], self.train_phase, 'conv5_1')
    conv_5_2 = self.conv_layer_with_bn(conv_5_1, [3, 3, 512, 512], self.train_phase, 'conv5_2')
    conv_5_3 = self.conv_layer_with_bn(conv_5_2, [3, 3, 512, 512], self.train_phase, 'conv5_3')
    pool_5, pool_5_argmax = self.pool_layer(conv_5_3)

    # First decoder
    unpool_5 = self.unpool_layer2x2(pool_5, pool_5_argmax, tf.shape(conv_5_3))
    deconv_5_3 = self.conv_layer_with_bn(unpool_5, [3, 3, 512, 512], self.train_phase, 'deconv5_3')
    deconv_5_2 = self.conv_layer_with_bn(deconv_5_3, [3, 3, 512, 512], self.train_phase, 'deconv5_2')
    deconv_5_1 = self.conv_layer_with_bn(deconv_5_2, [3, 3, 512, 512], self.train_phase, 'deconv5_1')

    # Second decoder
    unpool_4 = self.unpool_layer2x2(deconv_5_1, pool_4_argmax, tf.shape(conv_4_3))
    deconv_4_3 = self.conv_layer_with_bn(unpool_4, [3, 3, 512, 512], self.train_phase, 'deconv4_3')
    deconv_4_2 = self.conv_layer_with_bn(deconv_4_3, [3, 3, 512, 512], self.train_phase, 'deconv4_2')
    deconv_4_1 = self.conv_layer_with_bn(deconv_4_2, [3, 3, 512, 256], self.train_phase, 'deconv4_1')

    # Third decoder
    unpool_3 = self.unpool_layer2x2(deconv_4_1, pool_3_argmax, tf.shape(conv_3_3))
    deconv_3_3 = self.conv_layer_with_bn(unpool_3, [3, 3, 256, 256], self.train_phase, 'deconv3_3')
    deconv_3_2 = self.conv_layer_with_bn(deconv_3_3, [3, 3, 256, 256], self.train_phase, 'deconv3_2')
    deconv_3_1 = self.conv_layer_with_bn(deconv_3_2, [3, 3, 256, 128], self.train_phase, 'deconv3_1')

    # Fourth decoder
    unpool_2 = self.unpool_layer2x2(deconv_3_1, pool_2_argmax, tf.shape(conv_2_2))
    deconv_2_2 = self.conv_layer_with_bn(unpool_2, [3, 3, 128, 128], self.train_phase, 'deconv2_2')
    deconv_2_1 = self.conv_layer_with_bn(deconv_2_2, [3, 3, 128, 64], self.train_phase, 'deconv2_1')

    # Fifth decoder
    unpool_1 = self.unpool_layer2x2(deconv_2_1, pool_1_argmax, tf.shape(conv_1_2))
    deconv_1_2 = self.conv_layer_with_bn(unpool_1, [3, 3, 64, 64], self.train_phase, 'deconv1_2')
    deconv_1_1 = self.conv_layer_with_bn(deconv_1_2, [3, 3, 64, 32], self.train_phase, 'deconv1_1')

    # Produce class scores
    preds = self.conv_layer(deconv_1_1, [1, 1, 32, self.num_classes], 'preds')
    self.logits = tf.reshape(preds, (-1, self.num_classes))

    # Prepare network for training
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(expected, [-1]), logits=self.logits, name='x_entropy')
    self.loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')

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
    self.rate = learning_rate

    # Restore previous session if exists
    current_step = self.restore_session()

    dataset = DatasetReader()
    
    # Begin Training
    self.train_step = tf.train.AdamOptimizer(self.rate).minimize(self.loss)
    for i in range(current_step, num_iterations):

      # One training step
      image, ground_truth = dataset.next_train_pair()
      feed_dict = {self.x: [image], self.y: [ground_truth], self.train_phase: True}
      print('run train step: '+str(i))
      self.train_step.run(session=self.session, feed_dict=feed_dict)

      # Print loss every 10 iterations
      if i % 10 == 0:
        train_loss = self.session.run(self.loss, feed_dict=feed_dict)
        print("Step: %d, Train_loss:%g" % (i, train_loss))

      # Run against validation dataset for 100 iterations
      if i % 100 == 0:
        image, ground_truth = dataset.next_val_pair()
        feed_dict = {self.x: [image], self.y: [ground_truth], self.train_phase: True}
        val_loss = self.session.run(self.loss, feed_dict=feed_dict)
        val_accuracy = self.session.run(self.accuracy, feed_dict=feed_dict)
        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), val_loss))
        print("%s ---> Validation_accuracy: %g" % (datetime.datetime.now(), val_accuracy))

        self.logger.log("%s ---> Number of epochs: %g\n" % (datetime.datetime.now(), math.floor((i * batch_size)/bdr.num_train)))
        self.logger.log("%s ---> Number of iterations: %g\n" % (datetime.datetime.now(), i))
        self.logger.log("%s ---> Validation_loss: %g\n" % (datetime.datetime.now(), val_loss))
        self.logger.log("%s ---> Validation_accuracy: %g\n" % (datetime.datetime.now(), val_accuracy))

        # Save the model variables
        self.saver.save(self.session, self.checkpoint_directory + 'segnet', global_step = i)

  def test(self, learning_rate=1e-6):
    dataset = DatasetReader()
    image, ground_truth = dataset.next_test_pair() 
    feed_dict = {self.x: [image], self.y: [ground_truth], self.train_phase: False}
    prediction = self.session(self.logits, feed_dict=feed_dict)
    img = Image.fromarray(prediction, 'L')
    img.save('prediction.png')
