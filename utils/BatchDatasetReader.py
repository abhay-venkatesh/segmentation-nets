import random
import numpy as np
import cv2
from utils.ImageResizer import ImageResizer 

# TODO: Complete this
class BatchDatasetReader:
  ''' Helper class to SegNet that handles data reading, conversion 
    and all things related to data '''

  def __init__(self, directory, WIDTH, HEIGHT, current_step):
    self.training_data = open(directory + 'train.txt').readlines()
    self.train_index = current_step % 5000
    self.validation_data = open(directory + 'val.txt').readlines()
    self.test_data = open(directory + 'test.txt').readlines()
    self.text_index = 0
    self.WIDTH = WIDTH
    self.HEIGHT = HEIGHT
    self.epoch = 0

    # Resize images 
    print("Resizing dataset........ ")
    ground_truth_directory = directory + 'ground_truths/'
    ground_truth_output_directory = directory + 'ground_truths_resized/'
    ir = ImageResizer(ground_truth_directory, ground_truth_output_directory)
    ir.resize_ground_truths(WIDTH, HEIGHT)
    image_directory = directory + 'images/'
    images_output_directory = directory + 'images_resized/'
    ir = ImageResizer(image_directory, images_output_directory)
    ir.resize_images(WIDTH, HEIGHT)
    print("Finished resizing dataset. ")

  def shuffle_training_data(self):
    pass

  def next_trainining_batch(self, batch_size):
    images = []
    ground_truths = []

    for i in range(batch_size):
      pass

    return images, ground_truths



def main():
  bdr = BatchDatasetReader('./datasets/unreal_randomyaw/', 480, 320, 0)

if __name__ == "__main__":
  main()
