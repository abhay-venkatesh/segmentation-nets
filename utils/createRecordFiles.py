"""
Create train.txt and val.txt files
"""

import os
import math

def createTrainAndVal():
  directory = "../datasets/unreal_randomyaw/images/"
  file_count = next(os.walk(directory))[2]
  file_count = len(files)
  print(file_count)
  
  with open('../datasets/unreal_randomyaw/train.txt', 'w') as trainfile:
    with open('../datasets/unreal_randomyaw/val.txt', 'w') as valfile:

      numTrainingImages = math.floor(file_count * 0.7)
      numValidationImages = math.floor(file_count * 0.3)

      for i in range(numTrainingImages):
        trainfile.write('pic' + str(i) + ".png\n")

      for i in range(numValidationImages):
        valfile.write('pic' + str(i + numTrainingImages) + ".png\n")

def createTest():
  with open('test.txt', 'w') as testfile:

    numTestImages = 20
    for i in range(numTestImages):
      testfile.write('pic' + str(i) + ".png\n")

def main():
  createTrainAndVal()

if __name__ == "__main__":
  main()
