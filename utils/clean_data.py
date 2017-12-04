"""
Cleans and replaces missing images.
"""

import os
from shutil import copyfile

rootDir = './converted_data/'
numberList = []
for dirName, subDirList, fileList in os.walk(rootDir):
    for file in fileList:
        file = file.strip()
        if file != 'train.txt' and file != 'val.txt' and file[:3] == 'seg':
            fileNumber = file[3:]
            fileNumber = fileNumber[:-4]
            numberList.append(int(fileNumber))

counter = 0
numberList.sort()
numberList = set(numberList)
for num in numberList:
    if num != counter:
        fileName = './converted_data/seg' + str(num-2) + '.png'
        outFile = './converted_data/seg' + str(num-1) + '.png'
        copyfile(fileName, outFile)
        copyfile(fileName.replace('seg','pic'), outFile.replace('seg','pic'))
        print(num - 1)
        counter += 1
    counter += 1
    


