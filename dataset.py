import glob
import numpy as np
from PIL import Image
from skimage import io
from skimage.transform import resize
import tensorflow as tf
import os
from google.colab.patches import cv2_imshow

label_colormap = {
    "Void": (0, 0, 0, 255),
    "Crosswalk": (255, 0, 0, 255),
    "Double White": (0, 255, 0, 255),
    "Double Yellow": (0, 0, 255, 255),
    "Road Curb": (255, 255, 0, 255),
    "Single White": (255, 0, 255, 255),
    "Single Yellow": (0, 255, 255, 255)
}


id2label = [
    'Void',
    'Crosswalk',
    'Double White',
    'Double Yellow',
    'Road Curb',
    'Single White',
    'Single Yellow',
]

label2id = {label: id for id, label in enumerate(id2label)}
idcolormap = [label_colormap[label] for label in id2label]


class BDD100k():
    def __init__(self, dataDir, split, width, height):
        self.data = {}
        self.data['x_train'] = self.decode_images(
            dataDir+'images/', width, height)
        self.data['y_train'] = self.decode_labels(
            dataDir + 'labels/', width, height)
        self.data['x_train'], self.data['y_train'], self.data['x_val'], self.data['y_val'] = self.split_data(
            self.data['x_train'], self.data['y_train'], 0.1)
        self.data['weights'] = [0.21706227, 3.87922712, 9.64396687,
                                6.6800881,  3.99014264, 2.35461766, 7.59079911]
        self.data['n_classes'] = len(id2label)

    # def decode_images(self, dir, width, height):
    #     images = []
    #     for img in glob.glob(dir+'*.jpg'):
    #         img = tf.image.decode_jpeg(img, channels=3)
    #         img = tf.image.convert_image_dtype(img, tf.float32)
    #         img = tf.image.resize(img, [width, height])
    #         images.append(img)
    #     print("Loaded "+str(len(glob.glob(dir+'*.jpg')))+" Images")
    #     print(images[0].shape)
    #     images = np.asarray(images)
    #     print(images.shape)
    #     return images

    def decode_images(self, dir, width, height):
        images = []
        i = 0
        for img in glob.glob(dir+'*.jpg'):
            if i % 1000 == 0:
                print('Loaded ' + str(i)+' Images')
            i += 1
            img = Image.open(img).resize(
                (width, height), resample=Image.CUBIC)
            img = np.asarray(img, dtype=np.uint8)
            if i == 1:
                cv2_imshow(img)
            # img = np.asarray(img)
            # img = img/255.0
            # img = np.resize(img, (width, height))
            images.append(img)
            img = None
        print("Loaded "+str(len(glob.glob(dir+'*.jpg')))+" Images")
        print(images[0].shape)
        images = np.asarray(images)
        print(images.shape)
        return images

    # def decode_labels(self, dir, width, height):
    #     labels = []
    #     i = 0
    #     for img in glob.glob(dir+'*.png'):
    #         if i == 100:
    #             break
    #         i += 1
    #         img = tf.image.decode_png(img, channels=1)
    #         img = tf.image.resize(img, [width, height])
    #         labels.append(img)
    #     print("Loaded "+str(len(glob.glob(dir+'*.png')))+" Labels")
    #     labels = np.array(labels)
    #     return labels

    def decode_labels(self, dir, width, height):
        labels = []
        i = 0
        for img in glob.glob(dir+'*.png'):
            if i % 1000 == 0:
                print('Loaded ' + str(i)+' Images')
            i += 1
            img = Image.open(img).resize(
                (width, height), resample=Image.NEAREST)
            img = np.asarray(img, dtype=np.uint8)
            if i == 1:
                cv2_imshow(img)
            labels.append(img)
            img = None
        print("Loaded "+str(len(glob.glob(dir+'*.png')))+" Labels")
        labels = np.array(labels)
        return labels

    def split_data(self, x, y, ratio):
        n = len(x)
        n_val = int(round(n*ratio))
        x_val = x[:n_val]
        y_val = y[:n_val]
        x = x[n_val:]
        y = y[n_val:]
        return x, y, x_val, y_val

    def calculate_class_weights(self, labels):
        c = np.zeros(7)
        for img in glob.glob('/content/km10k/data/labels/*.png'):
            img = io.imread(img)
            ids, counts = np.unique(img, return_counts=True)
            for i in range(len(ids)):
                c[ids[i]] += counts[i]

        n_pixels = labels.size
        weights = np.zeros(7)
        for i in range(7):
            weights[i] = 1/np.log(1.10+c[i]/n_pixels)
        return weights
