"""
Reference: https://www.tensorflow.org/get_started/mnist/pros
"""

# Enable importing custom modules, specifically here - the mnist module for importing data
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
		W_conv1 = weight_variable([5, 5, 1, 32])
		b_conv1 = bias_variable([32])


