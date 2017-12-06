"""
Create train.txt, val.txt and test.txt files
"""

import os
import math

def create_files():
  directory = "../datasets/unreal_randomyaw/images/"
  file_count = next(os.walk(directory))[2]
  file_count = len(files)
  print(file_count)
  
  with open('../datasets/unreal_randomyaw/train.txt', 'w') as trainfile:
    with open('../datasets/unreal_randomyaw/val.txt', 'w') as valfile:
      with open('../datasets/unreal_randomyaw/test.txt', 'w') as testfile:

        numTrainingImages = math.floor(file_count * 0.7)
        numValidationImages = math.floor(file_count * 0.29)
        numTestImages = math.floor(file_count * 0.01)

        for i in range(numTrainingImages):
          trainfile.write('pic' + str(i) + ".png\n")

        for i in range(numValidationImages):
          valfile.write('pic' + str(i + numTrainingImages) + ".png\n")

        for i in range(numTestImages):
          testfile.write('pic' + str(i) + ".png\n")

def main():
  create_files()

if __name__ == "__main__":
  main()
