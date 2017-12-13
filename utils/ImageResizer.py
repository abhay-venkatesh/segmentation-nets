import os
import cv2

class ImageResizer:

  def __init__(self, input_directory, output_directory):
    # Quality assurance on directories
    assert os.path.exists(input_directory)
    if not os.path.exists(output_directory):
      os.makedirs(output_directory)

    # Store for usage
    self.input_directory = input_directory
    self.output_directory = output_directory

  def resize_ground_truths(self, WIDTH, HEIGHT):
    # Go over each file
    files = next(os.walk(self.input_directory))[2]
    for file in files:

      # Resize and write the file
      file_path = self.input_directory + file
      image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
      image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
      output_path = self.output_directory + file
      cv2.imwrite(output_path, image)

  def resize_images(self, WIDTH, HEIGHT):
    # Go over each file
    files = next(os.walk(self.input_directory))[2]
    for file in files:

      # Resize and write the file
      file_path = self.input_directory + file
      image = cv2.imread(file_path)
      image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
      output_path = self.output_directory + file
      cv2.imwrite(output_path, image)
      
def main():
  input_directory = '../datasets/unreal_randomyaw_27classes/ground_truths/'
  output_directory = '../datasets/unreal_randomyaw_27classes/ground_truths_resized/'
  imgRszr = ImageResizer(input_directory, output_directory)
  imgRszr.resize_ground_truths(480, 320)

if __name__ == "__main__":
  main()

       
