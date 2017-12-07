import random
import numpy as np
import cv2

class DatasetReader:
  ''' Helper class to SegNet that handles data reading, conversion 
      and all things related to data '''

  def __init__(self):
    self.training_data = open('./datasets/unreal_randomyaw/train.txt').readlines()
    self.validation_data = open('./datasets/unreal_randomyaw/val.txt').readlines()
    self.test_data = open('./datasets/unreal_randomyaw/test.txt').readlines()
    self.text_index = 0
    self.HEIGHT = 450
    self.WIDTH = 720

  def next_train_pair(self):
    # Load image
    image_directory = './datasets/unreal_randomyaw/images/'
    image_file = random.choice(self.training_data).rstrip()
    image = cv2.imread(image_directory + image_file)
    image = cv2.resize(image, (self.HEIGHT, self.WIDTH), interpolation=cv2.INTER_NEAREST)
    image = np.float32(image)

    # Load ground truth
    ground_truth_directory = './datasets/unreal_randomyaw/ground_truths/'
    ground_truth_file = image_file.replace('pic', 'seg')
    ground_truth = cv2.imread(ground_truth_directory + ground_truth_file, cv2.IMREAD_GRAYSCALE)
    ground_truth= cv2.resize(ground_truth, (self.HEIGHT, self.WIDTH), interpolation=cv2.INTER_NEAREST)
    ground_truth = ground_truth/8

    return image, ground_truth

  def next_val_pair(self):
    # Load image
    image_directory = './datasets/unreal_randomyaw/images/'
    image_file = random.choice(self.validation_data).rstrip()
    image = cv2.imread(image_directory + image_file)
    image = cv2.resize(image, (self.HEIGHT, self.WIDTH), interpolation=cv2.INTER_NEAREST)
    image = np.float32(image)

    # Load ground truth
    ground_truth_directory = './datasets/unreal_randomyaw/ground_truths/'
    ground_truth_file = image_file.replace('pic', 'seg')
    ground_truth = cv2.imread(ground_truth_directory + ground_truth_file, cv2.IMREAD_GRAYSCALE)
    ground_truth= cv2.resize(ground_truth, (self.HEIGHT, self.WIDTH), interpolation=cv2.INTER_NEAREST)
    ground_truth = ground_truth/8

    return image, ground_truth

  def next_test_pair(self):
    # Load image
    image_directory = './datasets/unreal_randomyaw/images/'
    image_file = self.test_data[self.test_index].rstrip()
    self.test_index += 1
    image = cv2.imread(image_directory + image_file)
    image = cv2.resize(image, (self.HEIGHT, self.WIDTH), interpolation=cv2.INTER_NEAREST)
    image = np.float32(image)

    # Load ground truth
    ground_truth_directory = './datasets/unreal_randomyaw/ground_truths/'
    ground_truth_file = image_file.replace('pic', 'seg')
    ground_truth = cv2.imread(ground_truth_directory + ground_truth_file, cv2.IMREAD_GRAYSCALE)
    ground_truth= cv2.resize(ground_truth, (self.HEIGHT, self.WIDTH), interpolation=cv2.INTER_NEAREST)
    ground_truth = ground_truth/8

    return image, ground_truth

  def next_train_batch(self):
    images = []
    ground_truths = []

    for i in range(2):
      image, ground_truth = self.next_train_pair()
      images.append(image)
      ground_truths.append(ground_truth)

    return images, ground_truths

  def next_val_batch(self):
    images = []
    ground_truths = []

    for i in range(2):
      image, ground_truth = self.next_val_pair()
      images.append(image)
      ground_truths.append(ground_truth)

    return images, ground_truths


def main():
    DatasetReader().next_train_pair()

if __name__ == "__main__":
    main()