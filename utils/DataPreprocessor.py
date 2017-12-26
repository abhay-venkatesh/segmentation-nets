import json
import cv2
import numpy as np
from ast import literal_eval
from shutil import copyfile
import os

class DataPreprocessor:

  def __init__(self):
    pass

  def match_color(self, object_mask, target_color, tolerance=15):
    """
      Args:
        object_mask: Ground truth semantic segmentation
        target_color: Color to match to
        tolerance: Match delta

      returns:
        match_region: The region that was matched by the color
    """
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

  def preprocess_reduced_classes(self):
    # Get color to class and class to number maps
    # Used in convert_image
    color2class = json.load(open('../dat/reducedColorsToClasses.json','r'))
    class2num = json.load(open('../dat/reducedClassesToInt.json','r'))
    color_map = {}
    for color in color2class:
      color_map[literal_eval(color)] = class2num[color2class[color]]
    self.preprocess(color_map)

  def preprocess_all_classes(self):
    # Get color to class and class to number maps
    # Used in convert_image
    color2class = json.load(open('../dat/finalColorsToClasses.json','r'))
    class2num = json.load(open('../dat/finalClassesToInt.json','r'))
    color_map = {}
    for color in color2class:
      color_map[literal_eval(color)] = class2num[color2class[color]]
    self.preprocess(color_map)

  def convert_image(self, dirName, fname, counter, color_map):
    ''' 
      Args:
        dirName: Name of the directory in which the files are placed
        fname: Name of the file to convert
        counter: Used to output an image with an index
        color_map: Colors to map to classes

      Writes the converted image to the data folder as well.
      Output image has pixel values from 0-27, 0 if no class
      and 1-27 are classes as described in the finalClassesToInt json file. 
    '''
    convertedPath = os.path.abspath(dirName)
    filePath = convertedPath + '\\' + fname
    img = cv2.imread(filePath)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    [m,n] = img.shape[:2]
    res = np.zeros((m,n))
    print("Working on" + filePath)
    for key in color_map:
      match_region=self.match_color(img,key)
      if not (match_region is None):
        res = (np.multiply(res, ~match_region)) + match_region*color_map[key]
    outfile = '../datasets/unreal_randomyaw/ground_truths/seg' + str(counter) + '.png' 
    print(outfile)
    cv2.imwrite(outfile,res*8)
    image_outfile = '../datasets/unreal_randomyaw/images/pic' + str(counter) + '.png'
    copyfile(filePath.replace('seg','pic'), image_outfile) 

  def preprocess(self, color_map):
    directory = '../datasets/unreal_randomyaw/' 
    if not os.path.exists(directory):
      os.makedirs(directory) 

    rootDir = "../../../../dataset_randomyaw/"
    counter = 0
    for dirName, subdirList, fileList in os.walk(rootDir):
      print('Found directory: %s' % dirName)
      for fname in fileList:

        if fname == 'seg0.png':
          self.convert_image(dirName, fname, counter, color_map)
          counter += 1

  def check_converted_image(self, dirName, fname):
    convertedPath = os.path.abspath(dirName)
    filePath = convertedPath + '\\' + fname
    img = cv2.imread(filePath,0)
    for i in range(len(img)):
      for j in range(len(img[0])):
        if img[i][j]/8 > 27:
          return False  

    return True

  def load_image(self, file_path, WIDTH, HEIGHT):
    """
      Args:
        file_path: Path to file to be loaded
        WIDTH:
        HEIGHT:

      Returns:
        A numpy array representing the image
    """
    image = cv2.imread(file_path)
    image = cv2.resize(image, (WIDTH, HEIGHT), interpolation=cv2.INTER_NEAREST)
    image = np.float32(image)
    return image

def main():
  dataProcessor = DataPreprocessor()
  dataProcessor.preprocess_reduced_classes() 

if __name__ == "__main__":
  main()
