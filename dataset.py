from utils import h52obj, obj2h5, idcolormap
import tensorflow as tf
import numpy as np
import random
import glob
import PIL
import sys
import os


class BDD100k():
    def __init__(self, datadir, shape, n_classes, state):
        self.imdir = os.path.join(datadir, state, 'images')
        self.lbdir = os.path.join(datadir, state, 'labels')
        self.n_classes = n_classes
        self.shape = shape
        self.x_files = []
        self.y_files = []
        self.weights = []

    def __len__(self):
        return len(self.x_files)

    def load_images(self):
        label_files = glob.glob(os.path.join(self.lbdir, "*.png"))
        file_ids = [os.path.basename(f).replace(
            "_L.png", ".png") for f in label_files]
        input_files = [os.path.join(self.imdir, file_id[:-3]+'jpg')
                       for file_id in file_ids]
        self.y_files = [np.string_(f) for f in label_files]
        self.x_files = [np.string_(f) for f in input_files]
        self.x_files.sort()
        self.y_files.sort()

    def preprocess_images(self, x, y, normalize=True, hot_label=True, weighted_label=True):
        width, height = self.shape
        n_samples = len(x)

        X = np.zeros([n_samples, height, width, 3], dtype=np.float32)
        if hot_label:
            Y = np.zeros([n_samples, height, width,
                          self.n_classes], dtype=np.float32)
        else:
            Y = np.zeros([n_samples, height, width], dtype=np.uint8)

        for i in range(n_samples):
            img = PIL.Image.open(x[i]).resize(
                (width, height), resample=PIL.Image.CUBIC)
            img = np.asarray(img, dtype=np.float32)

            label_img = PIL.Image.open(y[i]).resize(
                (width, height), resample=PIL.Image.NEAREST)
            label_img = np.asarray(label_img, dtype=np.float32)

            if normalize:
                img = np.divide(img, 255.0)

            label_img = self.seg2label(label_img)

            if hot_label:
                label_img = tf.keras.utils.to_categorical(
                    y=label_img,
                    num_classes=self.n_classes,
                    dtype='float32'
                )

            if weighted_label and hot_label:
                for j in range(self.n_classes):
                    label_img[label_img[:, :, j] == 1] *= self.weights[j]

            X[i] = img
            Y[i] = label_img
        return X, Y

    def calculate_class_weights(self, y):
        labels = y
        random.shuffle(labels)
        width, height = self.shape
        counts = np.zeros(self.n_classes, dtype=np.uint32)
        total = 0
        weights = np.zeros(self.n_classes, dtype=np.float32)
        np.set_printoptions(precision=4, suppress=True)
        for label_file in labels:
            label_img = PIL.Image.open(label_file).resize(
                (height, width), resample=PIL.Image.NEAREST)
            label_img = np.asarray(label_img, dtype=np.uint8)

            label_img = self.seg2label(label_img)

            ids, c = np.unique(label_img, return_counts=True)
            if total + sum(c) < 0:
                break
            total += sum(c)
            for id, cnt in zip(ids, c):
                if counts[id] + cnt < 0:
                    continue
                counts[id] += cnt
                weights[id] = (1 / counts[id])*(total)/2.0
            sys.stdout.write('\rCalculating Class Weights: '+str(weights))
            sys.stdout.flush()
        print('\n')
        self.weights = weights

    def seg2label(self, img):
        height, width, _ = img.shape
        label = np.zeros([height, width], dtype=np.uint8)
        for id in range(len(idcolormap)):
            label[np.all(img == np.array(idcolormap[id]), axis=2)] = id
        return label

    def save_weights(self, saveto):
        data = {'weights': self.weights}
        saveto = os.path.join(saveto, 'weights.h5')
        obj2h5(data, saveto)

    def load_weights(self, path):
        print('Loading existing weights...')
        try:
            path = os.path.join(path, 'weights.h5')
            data = h52obj(path)
            self.weights = data['weights']
            print('Weights loaded successfully!')
        except:
            print('Failed to load existing weights!')

    def shuffle(self, x, y):
        c = list(zip(x, y))
        random.shuffle(c)
        x, y = zip(*c)
        return x, y
