"""
Reference: https://www.tensorflow.org/get_started/mnist/pros
"""

import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir) 

# Get dataset as numpy arrays
from datasets.MNIST.mnist import read_data_sets 
mnist = read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

# None because that dimension will be determined by the batch size
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Initialize variables
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# Before using variables, initialize them
sess.run(tf.global_variables_initializer())

# Goal is to learn W for this regression model
y = tf.matmul(x,W) + b

# Compute loss
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

# Return an operation object that will allow us to perform gradient descent
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# Perform gradient descent
for _ in range(1000):
  batch = mnist.train.next_batch(100)

  # We must feed in data into the placeholders we had defined earlier
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

"""
Evaluate the model
"""

# First get a boolean array from comparing the ground truth to the predicted labels
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Convert the booleans to array of 0s and 1s
numbers_from_booleans = tf.cast(correct_prediction, tf.float32)

# Get the mean, which is the accuracy
accuracy = tf.reduce_mean(numbers_from_booleans)
print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
