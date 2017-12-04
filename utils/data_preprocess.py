import json
import cv2
import numpy as np
from ast import literal_eval
from shutil import copyfile
import os

# Get color to class and class to number maps
color2class = json.load(open('finalColorsToClasses.json','r'))
class2num = json.load(open('finalClassesToInt.json','r'))
color_map = {}
for color in color2class:
    color_map[literal_eval(color)] = class2num[color2class[color]]

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

def old_convert():
    with open('list','r') as f:
        for i,line in enumerate(f):
            itemname = line.strip()
            filename = '../../../dataset/batch1/' + itemname
            img = cv2.imread(filename)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
            [m,n] = img.shape[:2]
            res = np.zeros((m,n))

            print("Working on" + filename)

            for key in color_map:
                match_region=match_color(img,key)
                if not (match_region is None):
                    res = (np.multiply(res, ~match_region)) + match_region*color_map[key]

            outfile = 'converted_data/seg' + str(i) + '.png' 
            print(outfile)
            cv2.imwrite(outfile,res*8)

            copyfile(filename.replace('seg','pic'), outfile.replace('seg','pic'))

def new_convert():
    directory = './converted_data' 
    if not os.path.exists(directory):
        os.makedirs(directory) 

    rootDir = "../../../randomyaw_dataset/"
    counter = 0
    for dirName, subdirList, fileList in os.walk(rootDir):
        print('Found directory: %s' % dirName)
        for fname in fileList:

            if fname == 'seg0.png':
                convert_image(dirName, fname, counter)

            # Take multiple views for every 10th image
            '''
            if counter%10 == 0 and fname == 'seg1.png':
                counter += 1
                convert_image(dirName, fname, counter)
            '''

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

        
              

def convert_image(dirName, fname, counter):
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
    outfile = 'converted_data/seg' + str(counter) + '.png' 
    print(outfile)
    cv2.imwrite(outfile,res*8)
    copyfile(filePath.replace('seg','pic'), outfile.replace('seg','pic')) 

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
    # old_convert()
    # new_convert()
    # convert_for_test()
    print(check_converted_image('./converted_data', 'seg10.png'))


if __name__ == "__main__":
    main()
