import os
import math

class RecordFileGenerator:

  def __init__(self):
    # Get size of dataset
    directory = "../datasets/unreal_randomyaw/images/"
    files = next(os.walk(directory))[2]
    self.dataset_size = len(files)

  def create_files(self):
    ''' Create train.txt, val.txt and test.txt files. '''
    # Open files
    with open('../datasets/unreal_randomyaw/train.txt', 'w') as trainfile:
      with open('../datasets/unreal_randomyaw/val.txt', 'w') as valfile:
        with open('../datasets/unreal_randomyaw/test.txt', 'w') as testfile:

          # Get image counts
          numTrainingImages = math.floor(self.dataset_size * 0.7)
          numValidationImages = math.floor(self.dataset_size * 0.29)
          numTestImages = math.floor(self.dataset_size * 0.01)

          # Write the images
          for i in range(numTrainingImages):
            trainfile.write('pic' + str(i) + ".png\n")
          for i in range(numValidationImages):
            valfile.write('pic' + str(i + numTrainingImages) + ".png\n")
          for i in range(numTestImages):
            testfile.write('pic' + str(i) + ".png\n")

def main():
  RecordFileGenerator().create_files()

if __name__ == "__main__":
  main()
