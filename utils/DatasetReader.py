import random
import numpy as np
import cv2

class DatasetReader:
  ''' Helper class to SegNet that handles data reading, conversion 
      and all things related to data. Non-Batch Version. '''

  def __init__(self, WIDTH, HEIGHT, dataset_directory):
    self.dataset_directory = dataset_directory
    training_data_file = dataset_directory + 'train.txt'
    validation_data_file = dataset_directory + 'val.txt'
    test_data_file = dataset_directory + 'test.txt'
    self.training_data = open('./datasets/unreal_randomyaw/train.txt').readlines()
    self.validation_data = open('./datasets/unreal_randomyaw/val.txt').readlines()
    self.test_data = open('./datasets/unreal_randomyaw/test.txt').readlines()
    self.test_data_size = len(self.test_data)
    self.test_index = 0
    self.WIDTH = WIDTH
    self.HEIGHT = HEIGHT

  def next_train_pair(self):
    # Load image
    image_directory = self.dataset_directory + 'images/'
    image_file = random.choice(self.training_data).rstrip()
    image = cv2.imread(image_directory + image_file)
    image = cv2.resize(image, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
    image = np.float32(image)

    # Load ground truth
    ground_truth_directory = self.dataset_directory + 'ground_truths/'
    ground_truth_file = image_file.replace('pic', 'seg')
    ground_truth = cv2.imread(ground_truth_directory + ground_truth_file, cv2.IMREAD_GRAYSCALE)
    ground_truth= cv2.resize(ground_truth, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
    ground_truth = ground_truth/8

    return image, ground_truth

  def next_val_pair(self):
    # Load image
    image_directory = self.dataset_directory + 'images/'
    image_file = random.choice(self.validation_data).rstrip()
    image = cv2.imread(image_directory + image_file)
    image = cv2.resize(image, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
    image = np.float32(image)

    # Load ground truth
    ground_truth_directory = self.dataset_directory + 'ground_truths/'
    ground_truth_file = image_file.replace('pic', 'seg')
    ground_truth = cv2.imread(ground_truth_directory + ground_truth_file, cv2.IMREAD_GRAYSCALE)
    ground_truth= cv2.resize(ground_truth, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
    ground_truth = ground_truth/8

    return image, ground_truth

  def next_test_pair(self):
    # Load image
    image_directory = self.dataset_directory + 'images/'
    image_file = self.test_data[self.test_index].rstrip()
    self.test_index += 1
    image = cv2.imread(image_directory + image_file)
    image = cv2.resize(image, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
    image = np.float32(image)

    # Load ground truth
    ground_truth_directory = self.dataset_directory + 'ground_truths/'
    ground_truth_file = image_file.replace('pic', 'seg')
    ground_truth = cv2.imread(ground_truth_directory + ground_truth_file, cv2.IMREAD_GRAYSCALE)
    ground_truth= cv2.resize(ground_truth, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
    ground_truth = ground_truth/8

    return image, ground_truth

def main():
    pass
if __name__ == "__main__":
    main()
