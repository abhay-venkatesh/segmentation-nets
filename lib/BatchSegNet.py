# Import modules for building the network
import tensorflow as tf
import numpy as np
import scipy.io
from math import ceil
import cv2
from utils.DatasetReader import DatasetReader
from PIL import Image
import datetime
import os

class BatchSegNet:
  ''' Network described by,
  https://arxiv.org/pdf/1505.04366v1.pdf
  and https://arxiv.org/pdf/1505.07293.pdf
  and https://arxiv.org/pdf/1511.00561.pdf '''

  def load_vgg_weights(self):
    """ Use the VGG model trained on
      imagent dataset as a starting point for training """
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
    return tf.nn.max_pool_with_argmax(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

  def unpool(self, pool, ind, ksize=[1, 2, 2, 1], scope='unpool'):
    """
       Unpooling layer after max_pool_with_argmax.
       Args:
           pool:   max pooled output tensor
           ind:      argmax indices
           ksize:     ksize is the same as for the pool
       Return:
           unpool:    unpooling tensor
    """
    with tf.variable_scope(scope):
      input_shape =  tf.shape(pool)
      output_shape = [input_shape[0], input_shape[1] * ksize[1], input_shape[2] * ksize[2], input_shape[3]]

      flat_input_size = tf.cumprod(input_shape)[-1]
      flat_output_shape = tf.stack([output_shape[0], output_shape[1] * output_shape[2] * output_shape[3]])

      pool_ = tf.reshape(pool, tf.stack([flat_input_size]))
      batch_range = tf.reshape(tf.range(tf.cast(output_shape[0], tf.int64), dtype=ind.dtype), 
                                        shape=tf.stack([input_shape[0], 1, 1, 1]))
      b = tf.ones_like(ind) * batch_range
      b = tf.reshape(b, tf.stack([flat_input_size, 1]))
      ind_ = tf.reshape(ind, tf.stack([flat_input_size, 1]))
      ind_ = tf.concat([b, ind_], 1)

      ret = tf.scatter_nd(ind_, pool_, shape=tf.cast(flat_output_shape, tf.int64))
      ret = tf.reshape(ret, tf.stack(output_shape))
      return ret

  # Implementation idea from: https://github.com/tensorflow/tensorflow/issues/2169
  def unravel_argmax(self, argmax, shape):
    output_list = []
    output_list.append(argmax // (shape[2] * shape[3]))
    output_list.append(argmax % (shape[2] * shape[3]) // shape[3])
    return tf.stack(output_list)
  
  def unpool_layer2x2_batch(self, x, argmax):
    '''
    Args:
        x: 4D tensor of shape [batch_size x height x width x channels]
        argmax: A Tensor of type Targmax. 4-D. The flattened indices of the max
        values chosen for each output.
    Return:
        4D output tensor of shape [batch_size x 2*height x 2*width x channels]
    '''
    bottom_shape = tf.shape(x)
    top_shape = [bottom_shape[0], bottom_shape[1]*2, bottom_shape[2]*2, bottom_shape[3]]

    batch_size = top_shape[0]
    height = top_shape[1]
    width = top_shape[2]
    channels = top_shape[3]

    argmax_shape = tf.to_int64([batch_size, height, width, channels])
    argmax = self.unravel_argmax(argmax, argmax_shape)

    t1 = tf.to_int64(tf.range(channels))
    t1 = tf.tile(t1, [batch_size*(width//2)*(height//2)])
    t1 = tf.reshape(t1, [-1, channels])
    t1 = tf.transpose(t1, perm=[1, 0])
    t1 = tf.reshape(t1, [channels, batch_size, height//2, width//2, 1])
    t1 = tf.transpose(t1, perm=[1, 0, 2, 3, 4])

    t2 = tf.to_int64(tf.range(batch_size))
    t2 = tf.tile(t2, [channels*(width//2)*(height//2)])
    t2 = tf.reshape(t2, [-1, batch_size])
    t2 = tf.transpose(t2, perm=[1, 0])
    t2 = tf.reshape(t2, [batch_size, channels, height//2, width//2, 1])

    t3 = tf.transpose(argmax, perm=[1, 4, 2, 3, 0])

    t = tf.concat([t2, t3, t1], 4)
    indices = tf.reshape(t, [(height//2)*(width//2)*channels*batch_size, 4])

    x1 = tf.transpose(x, perm=[0, 3, 1, 2])
    values = tf.reshape(x1, [-1])

    delta = tf.SparseTensor(indices, values, tf.to_int64(top_shape))
    return tf.sparse_tensor_to_dense(tf.sparse_reorder(delta))
        
  def conv_layer(self, x, W_shape, b_shape, name, padding='SAME'):
    # Pass b_shape as list because need the object to be iterable for the constant initializer
    W, b = self.vgg_weight_and_bias(name, W_shape, [b_shape])

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
    self.x = tf.placeholder(tf.float32, shape=(None, None, None, 3))
    self.y = tf.placeholder(tf.int64, shape=(None, None, None))
    expected = tf.expand_dims(self.y, -1)
    self.rate = tf.placeholder(tf.float32, shape=[])

    # First encoder
    conv_1_1 = self.conv_layer(self.x, [3, 3, 3, 64], 64, 'conv1_1')
    conv_1_2 = self.conv_layer(conv_1_1, [3, 3, 64, 64], 64, 'conv1_2')
    pool_1, pool_1_argmax = self.pool_layer(conv_1_2)

    # Second encoder
    conv_2_1 = self.conv_layer(pool_1, [3, 3, 64, 128], 128, 'conv2_1')
    conv_2_2 = self.conv_layer(conv_2_1, [3, 3, 128, 128], 128, 'conv2_2')
    pool_2, pool_2_argmax = self.pool_layer(conv_2_2)

    # Third encoder
    conv_3_1 = self.conv_layer(pool_2, [3, 3, 128, 256], 256, 'conv3_1')
    conv_3_2 = self.conv_layer(conv_3_1, [3, 3, 256, 256], 256, 'conv3_2')
    conv_3_3 = self.conv_layer(conv_3_2, [3, 3, 256, 256], 256, 'conv3_3')
    pool_3, pool_3_argmax = self.pool_layer(conv_3_3)

    # Fourth encoder
    conv_4_1 = self.conv_layer(pool_3, [3, 3, 256, 512], 512, 'conv4_1')
    conv_4_2 = self.conv_layer(conv_4_1, [3, 3, 512, 512], 512, 'conv4_2')
    conv_4_3 = self.conv_layer(conv_4_2, [3, 3, 512, 512], 512, 'conv4_3')
    pool_4, pool_4_argmax = self.pool_layer(conv_4_3)

    # Fifth encoder
    conv_5_1 = self.conv_layer(pool_4, [3, 3, 512, 512], 512, 'conv5_1')
    conv_5_2 = self.conv_layer(conv_5_1, [3, 3, 512, 512], 512, 'conv5_2')
    conv_5_3 = self.conv_layer(conv_5_2, [3, 3, 512, 512], 512, 'conv5_3')
    pool_5, pool_5_argmax = self.pool_layer(conv_5_3)

    # First decoder
    unpool_5 = self.unpool(pool_5, pool_5_argmax)
    deconv_5_3 = self.deconv_layer(unpool_5, [3, 3, 512, 512], 512, 'deconv5_3')
    deconv_5_2 = self.deconv_layer(deconv_5_3, [3, 3, 512, 512], 512, 'deconv5_2')
    deconv_5_1 = self.deconv_layer(deconv_5_2, [3, 3, 512, 512], 512, 'deconv5_1')

    # Second decoder
    unpool_4 = self.unpool(deconv_5_1, pool_4_argmax)
    deconv_4_3 = self.deconv_layer(unpool_4, [3, 3, 512, 512], 512, 'deconv4_3')
    deconv_4_2 = self.deconv_layer(deconv_4_3, [3, 3, 512, 512], 512, 'deconv4_2')
    deconv_4_1 = self.deconv_layer(deconv_4_2, [3, 3, 256, 512], 256, 'deconv4_1')

    # Third decoder
    unpool_3 = self.unpool(deconv_4_1, pool_3_argmax)
    deconv_3_3 = self.deconv_layer(unpool_3, [3, 3, 256, 256], 256, 'deconv3_3')
    deconv_3_2 = self.deconv_layer(deconv_3_3, [3, 3, 256, 256], 256, 'deconv3_2')
    deconv_3_1 = self.deconv_layer(deconv_3_2, [3, 3, 128, 256], 128, 'deconv3_1')

    # Fourth decoder
    unpool_2 = self.unpool(deconv_3_1, pool_2_argmax)
    deconv_2_2 = self.deconv_layer(unpool_2, [3, 3, 128, 128], 128, 'deconv2_2')
    deconv_2_1 = self.deconv_layer(deconv_2_2, [3, 3, 64, 128], 64, 'deconv2_1')

    # Fifth decoder
    unpool_1 = self.unpool(deconv_2_1, pool_1_argmax)
    deconv_1_2 = self.deconv_layer(unpool_1, [3, 3, 64, 64], 64, 'deconv1_2')
    deconv_1_1 = self.deconv_layer(deconv_1_2, [3, 3, 32, 64], 32, 'deconv1_1')

    # Produce class scores
    score_1 = self.deconv_layer(deconv_1_1, [1, 1, 28, 32], 28, 'score_1')
    logits = tf.reshape(score_1, (-1, 28))

    # Prepare network for training
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=tf.reshape(expected, [-1]), logits=logits, name='x_entropy')
    self.loss = tf.reduce_mean(cross_entropy, name='x_entropy_mean')
    self.train_step = tf.train.AdamOptimizer(self.rate).minimize(self.loss)

    # Metrics
    self.prediction = tf.argmax(tf.reshape(tf.nn.softmax(logits), tf.shape(score_1)), axis=3)
    self.accuracy = tf.reduce_sum(tf.pow(self.prediction - tf.squeeze(expected), 2))

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

  
  def train(self, num_iterations=10000, learning_rate=1e-6, batch_size=5):
    # Restore previous session if exists
    current_step = self.restore_session()

    dataset = DatasetReader()
    

    # Count number of items trained on
    count = 0
    count += (current_step * batch_size)

    # Begin Training
    for i in range(current_step, num_iterations):

      # One training step
      images, ground_truths = dataset.next_train_batch(batch_size)
      feed_dict = {self.x: images, self.y: ground_truths, self.rate: learning_rate}
      print('run train step: ' + str(i))
      self.train_step.run(session=self.session, feed_dict=feed_dict)

      # Print loss every 10 iterations
      if i % 10 == 0:
        train_loss = self.session.run(self.loss, feed_dict=feed_dict)
        print("Step: %d, Train_loss:%g" % (i, train_loss))

      # Run against validation dataset for 100 iterations
      if i % 100 == 0:
        images, ground_truths = dataset.next_val_batch(batch_size)
        feed_dict = {self.x: images, self.y: ground_truths, self.rate: learning_rate}
        val_loss = self.session.run(self.loss, feed_dict=feed_dict)
        val_accuracy = self.session.run(self.accuracy, feed_dict=feed_dict)
        print("%s ---> Validation_loss: %g" % (datetime.datetime.now(), val_loss))
        print("%s ---> Validation_accuracy: %g" % (datetime.datetime.now(), val_accuracy))

        # Save the model variables
        self.saver.save(self.session, self.checkpoint_directory + 'segnet', global_step = i)

      count += batch_size
      if count % 5000 == 0:
        print("---- COMPLETED " + str(count/5000) + " EPOCH(S) ----")