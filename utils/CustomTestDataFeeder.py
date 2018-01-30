import os
import cv2
import numpy as np

class CustomTestDataFeeder:
  def __init__(self, WIDTH, HEIGHT, dataset_directory):
    self.WIDTH = WIDTH
    self.HEIGHT = HEIGHT
    self.dataset_directory = dataset_directory
    self.image_directory = dataset_directory + 'images/'

    # Files is an array with all the files in the directory.
    files = next(os.walk(self.image_directory))[2]

    self.dataset_size = len(files)
    self.test_index = 0

  def next_test_image(self):
    """ 
      Args:

      Returns:
        Image, Ground Truth pair

      On Error:
        Returns None, None
    """
    if self.test_index < self.dataset_size:
      # Load image, resize it and produce a numpy array
      image_directory = self.dataset_directory + 'images/'
      image_file = 'pic' + str(self.test_index) + '.png'
      print("Reading ... " + image_directory + image_file)
      image = cv2.imread(image_directory + image_file)
      image = cv2.resize(image, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
      image = np.float32(image)

      # Load ground truth image, resize it and produce a numpy array
      ground_truth_directory = self.dataset_directory + 'ground_truths/'
      ground_truth_file = 'seg' + str(self.test_index) + '.png'
      ground_truth = cv2.imread(ground_truth_directory + ground_truth_file, cv2.IMREAD_GRAYSCALE)
      ground_truth= cv2.resize(ground_truth, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
      ground_truth = ground_truth/8

      self.test_index += 1

      return image, ground_truth
    else: 
      return None, None 
