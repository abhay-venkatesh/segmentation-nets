import numpy as np
import cv2
import scipy.misc as misc
import tensorflow as tf

HEIGHT = 450
WIDTH = 720
MEAN = [104.00699, 116.66877, 122.67892]

id2color={"1": [143, 47, 47], "2": [191, 127, 0], "3": [223, 175, 31], "4": [95, 255, 31], "5": [175, 47, 15], "6": [175, 111, 223], "7": [223, 47, 47], "8": [159, 15, 127], "9": [79, 111, 63], "10": [31, 111, 223], "11": [95, 191, 175], "12": [223, 79, 255], "13": [79, 79, 79], "14": [31, 31, 223], "15": [191, 47, 255], "16": [79, 255, 63], "17": [223, 175, 63], "18": [47, 207, 95], "19": [223, 207, 63], "20": [95, 111, 127], "21": [63, 159, 191], "22": [0, 143, 79], "23": [15, 15, 255], "24": [159, 95, 223], "25": [95, 111, 15], "26": [191, 95, 0], "27": [127, 191, 207]}

def getColorSeg(seg):
    seg_color = np.zeros((HEIGHT,WIDTH,3),dtype=np.int32)

    for i in range(HEIGHT):
        for j in range(WIDTH):
            if seg[i][j]==0:
                seg_color[i][j] = [0,0,0]
            else:
                seg_color[i][j] = id2color[str(seg[i][j])]
    return seg_color

def processData(data):
    red, green, blue = tf.split(data, 3, 3)
    data = tf.concat([blue - MEAN[0], green - MEAN[1],red - MEAN[2]], 3)
    return data

def readTrainDataset(data_dir):
    with open('%s/train.txt'%data_dir) as f:
        train_records = f.read().split('\n')[:-1]
    with open('%s/val.txt'%data_dir) as f:
        val_records = f.read().split('\n')[:-1]
    return train_records, val_records

def readTestDataset(data_dir, theta):
    with open('%s/test_%d.txt'%(data_dir,theta)) as f:
        test_records = f.read().split('\n')[:-1]
    return test_records

class BatchDataset:
    batch_offset = 0
    epochs_completed = 0

    def __init__(self, base_dir,records_list, image_options={}):
        self.files = records_list
        self.image_options = image_options
        self.base_dir = base_dir
        self._read_images()

    def _read_images(self):
        self.images = np.array([self._transform("%s/%s"%(self.base_dir,filename),True) for filename in self.files],np.float32)
    	self.labels = []
    	for filename in self.files:
    	    label = self._transform("%s/%s"%(self.base_dir,filename.replace('pic','seg')))
            self.labels.append(np.array(np.expand_dims(label/8, axis=3),np.int32))
        self.labels = np.array(self.labels, np.int32)

    def _transform(self, filename, specify_mode=False):
    	if specify_mode:
            image = misc.imread(filename,mode='RGB')
    	else:
    	    image = misc.imread(filename)

        if self.image_options.get("resize", False) and self.image_options["resize"]:
#            resize_image = cv2.resize(image, (0,0), fx = 0.5, fy = 0.5)
#            resize_size = int(self.image_options["resize_size"])
            resize_image = misc.imresize(image,[HEIGHT, WIDTH], interp='nearest')
        else:
            resize_image = image

        return np.array(resize_image)

    def get_records(self):
        return self.images, self.labels

    def reset_batch_offset(self, offset=0):
        self.batch_offset = offset

    def next_batch(self, batch_size):
        start = self.batch_offset
        self.batch_offset += batch_size
        if self.batch_offset > self.images.shape[0]:
            # Finished epoch
            self.epochs_completed += 1
            print("****************** Epochs completed: " + str(self.epochs_completed) + "******************")
            # Shuffle the data
            perm = np.arange(self.images.shape[0])
            np.random.shuffle(perm)
            self.images = self.images[perm]
            self.labels = self.labels[perm]
            # Start next epoch
            start = 0
            self.batch_offset = batch_size

        end = self.batch_offset
        return self.images[start:end], self.labels[start:end]

    def get_random_batch(self, batch_size):
        indexes = np.random.randint(0, self.images.shape[0], size=[batch_size]).tolist()
        return self.images[indexes], self.labels[indexes]
