import random
import numpy as np
import cv2

class OutsideDataFeeder:
  ''' Helper class to SegNet that handles data reading, conversion 
      and all things related to data '''

  def __init__(self, WIDTH, HEIGHT, dataset_directory):
    self.dataset_directory = dataset_directory
    self.test_data = open(dataset_directory + 'test.txt').readlines()
    self.test_data_size = len(self.test_data)
    self.test_index = 0
    self.WIDTH = WIDTH
    self.HEIGHT = HEIGHT

  def next_test_image(self):
    # Load image
    image_directory = self.dataset_directory + 'images/'
    image_file = self.test_data[self.test_index].rstrip()
    self.test_index += 1
    image = cv2.imread(image_directory + image_file)
    image = cv2.resize(image, (self.WIDTH, self.HEIGHT), interpolation=cv2.INTER_NEAREST)
    image = np.float32(image)
    return image

def main():
    OutsideDataFeeder().next_test_image()

if __name__ == "__main__":
    main()
