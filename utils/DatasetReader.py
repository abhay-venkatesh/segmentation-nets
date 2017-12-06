import random
import numpy as np
import cv2

# Helper class to SegNet that handles data reading, conversion and all things related to data 
class DatasetReader:

  self.text_index = 0

  def __init__(self):
    self.training_data = open('./datasets/unreal_randomyaw/train.txt').readlines()
    self.validation_data = open('./datasets/unreal_randomyaw/val.txt').readlines()
    self.test_data = open('./datasets/unreal_randomyaw/test.txt').readlines()

  def next_train_pair(self):

    # Load image
    image_directory = './datasets/unreal_randomyaw/images/'
    image_file = random.choice(self.training_data)
    image = np.float32(cv2.imread(image_directory + image_file))

    # Load ground truth
    ground_truth_directory = './datasets/unreal_randomyaw/ground_truths/'
    ground_truth_file = image_file.replace('pic', 'seg')
    ground_truth = cv2.imread(ground_truth_directory + ground_truth_file, cv2.IMREAD_GRAYSCALE)
    ground_truth = ground_truth/8

    return image, ground_truth

  def next_val_pair(self):

    # Load image
    image_directory = './datasets/unreal_randomyaw/images/'
    image_file = random.choice(self.validation_data)
    image = np.float32(cv2.imread(image_directory + image_file))

    # Load ground truth
    ground_truth_directory = './datasets/unreal_randomyaw/ground_truths/'
    ground_truth_file = image_file.replace('pic', 'seg')
    ground_truth = cv2.imread(ground_truth_directory + ground_truth_file, cv2.IMREAD_GRAYSCALE)
    ground_truth = ground_truth/8

    return image, ground_truth

  def next_test_pair(self):

    # Load image
    image_directory = './datasets/unreal_randomyaw/images/'
    image_file = self.test_data[self.test_index]
    self.test_index += 1
    image = np.float32(cv2.imread(image_directory + image_file))

    # Load ground truth
    ground_truth_directory = './datasets/unreal_randomyaw/ground_truths/'
    ground_truth_file = image_file.replace('pic', 'seg')
    ground_truth = cv2.imread(ground_truth_directory + ground_truth_file, cv2.IMREAD_GRAYSCALE)
    ground_truth = ground_truth/8

    return image, ground_truth