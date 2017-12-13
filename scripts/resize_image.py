import random
import numpy as np
import cv2
from matplotlib import pyplot as plt
import scipy.misc as misc

def next_train_pair():
  training_data = open('../datasets/unreal_randomyaw/train.txt').readlines()
  HEIGHT = 320
  WIDTH = 480

  # Load image
  image_directory = '../datasets/unreal_randomyaw/images/'
  image_file = random.choice(training_data).rstrip()
  image = cv2.imread(image_directory + image_file)
  image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
  
  image = np.float32(image)

  ground_truth_directory = '../datasets/unreal_randomyaw/ground_truths/'
  ground_truth_file = image_file.replace('pic', 'seg')
  ground_truth = cv2.imread(ground_truth_directory + ground_truth_file, cv2.IMREAD_GRAYSCALE)
  ground_truth= cv2.resize(ground_truth, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
  ground_truth = ground_truth
  plt.imshow(ground_truth, cmap='gray')
  plt.show()

def main():
  next_train_pair()

if __name__ == "__main__":
  main()