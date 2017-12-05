"""
Reference: https://www.tensorflow.org/get_started/mnist/pros
"""

# Enable importing custom modules, specifically here - the mnist module for importing MNIST dataset
import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

# Get dataset as numpy arrays
from datasets.MNIST.mnist import read_data_sets 
mnist = read_data_sets('MNIST_data', one_hot=True)

class ConvNetClassifier:

	def __init__(self):
		self.build()

	"""
	Functions to initialize weights and biases.

	TODO: Try using other distributions to initalize the weights.
	"""
	def weight_variable(self, shape):
	  initial = tf.truncated_normal(shape, stddev=0.1)
	  return tf.Variable(initial)

	def bias_variable(self, shape):
	  initial = tf.constant(0.1, shape=shape)
	  return tf.Variable(initial)

	def conv2d(self, x, W):
	  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(self, x):
	  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

	def build(self):

		# Input layer
		x_image = tf.reshape(x, [-1, 28, 28, 1])

		# First convolutional layer
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
		h_pool1 = max_pool_2x2(h_conv1)
	
		# Second convolutional layer
		W_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
		h_pool2 = max_pool_2x2(h_conv2)

		# Fully connected layer
		W_fc1 = weight_variable([7 * 7 * 64, 1024])
		b_fc1 = bias_variable([1024])
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

		# Readout layer
		W_fc2 = weight_variable([1024, 10])
		b_fc2 = bias_variable([10])
		y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2


