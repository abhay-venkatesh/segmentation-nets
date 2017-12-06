import json
import cv2
import numpy as np
from ast import literal_eval
from shutil import copyfile
import os

# Match an i
def match_color(object_mask, target_color, tolerance=3):
  match_region = np.ones(object_mask.shape[0:2], dtype=bool)
  for c in range(3):
    min_val = target_color[c] - tolerance
    max_val = target_color[c] + tolerance
    channel_region = (object_mask[:,:,c] >= min_val) & (object_mask[:,:,c] <= max_val)
    match_region &= channel_region

    if match_region.sum() != 0:
      return match_region
    else:
      return None

# Get color to class and class to number maps
# Used in convert_image
color2class = json.load(open('finalColorsToClasses.json','r'))
class2num = json.load(open('finalClassesToInt.json','r'))
color_map = {}
for color in color2class:
  color_map[literal_eval(color)] = class2num[color2class[color]]

def convert_image(dirName, fname, counter):
  ''' Takes a directory name, file name and a counter and 
  converts the corresponding colored segmentation to grayscale.

  Writes the converted image to the data folder as well.
  Output image has pixel values from 0-27, 0 if no class
  and 1-27 are classes as described in the finalClassesToInt json file. '''
  convertedPath = os.path.abspath(dirName)
  filePath = convertedPath + '\\' + fname
  img = cv2.imread(filePath)
  img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
  [m,n] = img.shape[:2]
  res = np.zeros((m,n))
  print("Working on" + filePath)
  for key in color_map:
    match_region=match_color(img,key)
    if not (match_region is None):
      res = (np.multiply(res, ~match_region)) + match_region*color_map[key]
  outfile = '../datasets/unreal_randomyaw/ground_truths/seg' + str(counter) + '.png' 
  print(outfile)
  cv2.imwrite(outfile,res*8)
  image_outfile = '../datasets/unreal_randomyaw/images/pic' + str(counter) + '.png'
  copyfile(filePath.replace('seg','pic'), image_outfile) 

def preprocess():
  directory = '../datasets/unreal_randomyaw/' 
  if not os.path.exists(directory):
    os.makedirs(directory) 

  rootDir = "../../../../dataset_randomyaw/"
  counter = 0
  for dirName, subdirList, fileList in os.walk(rootDir):
    print('Found directory: %s' % dirName)
    for fname in fileList:

      if fname == 'seg0.png':
        convert_image(dirName, fname, counter)
        counter += 1

def convert_for_test():
  directory = './converted_data' 
  if not os.path.exists(directory):
    os.makedirs(directory) 

  rootDir = "../../../dataset/batch8/round3"
  counter = 0
  for dirName, subdirList, fileList in os.walk(rootDir):
    for fname in fileList:
      if fname[:3] == 'seg':
        convertedPath = os.path.abspath(dirName)
        filePath = convertedPath + '\\' + fname
        img = cv2.imread(filePath)
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        [m,n] = img.shape[:2]
        res = np.zeros((m,n))
        print("Working on" + filePath)
        for key in color_map:
          match_region=match_color(img,key)
          if not (match_region is None):
            res = (np.multiply(res, ~match_region)) + match_region*color_map[key]
        outfile = 'test_data/seg' + str(counter) + '.png' 
        print(outfile)
        cv2.imwrite(outfile,res*8)
        copyfile(filePath.replace('seg','pic'), outfile.replace('seg','pic')) 
        counter += 1         

def check_converted_image(dirName, fname):
  convertedPath = os.path.abspath(dirName)
  filePath = convertedPath + '\\' + fname
  img = cv2.imread(filePath,0)
  for i in range(len(img)):
    for j in range(len(img[0])):
      if img[i][j]/8 > 27:
        return False  

  return True

def main():
  preprocess() 


if __name__ == "__main__":
  main()
