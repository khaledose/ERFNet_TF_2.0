from __future__ import print_function, division, unicode_literals
import glob
import os
import PIL.Image
import numpy as np
import h5py
import random
import sys
import tensorflow as tf
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
    def __init__(self, file, data, train_method, action):
        self.data_dir = file['datadir']
        self.h5_dataset = file['data'] 
        self.h5_viz = file['viz'] 
        self.h5_weights = file['weights']
        self.h5_history = file['history']  
        self.h5_iou = file['iou']
        self.width, self.height = data['width'], data['height']
        self.shape = [self.width, self.height]
        self.n_channels = 3
        self.label_chanel_axis = False
        self.n_classes  = data['n_classes']
        if not os.path.isfile(self.h5_dataset):
            self.data_files = self.prepare_data_files(data)
        self.data_files = h52obj(self.h5_dataset)
        self.train_samples = len(self.data_files['x_train'])
        self.val_samples = len(self.data_files['x_val'])

        print('Preparing Data')
        if not os.path.isfile(self.h5_weights):
            self.calculate_class_weights(data)
        self.weights = h52obj(self.h5_weights)

        if not os.path.isfile(self.h5_viz):
            self.prepare_viz_data()
        self.viz = h52obj(self.h5_viz)

        if not os.path.isfile(self.h5_history):
            self.prepare_history()
        self.history = h52obj(self.h5_history)

        if action['train_iou'] and action['val_iou']:
            if not os.path.isfile(self.h5_iou):
                self.prepare_iou_data()
            self.iou_data = h52obj(self.h5_iou)

        if train_method['ram']:
            self.inputs = self.prepare_batch('train')

        print('Done')

    def prepare_data_files(self, data):
        data_files = {}
        train_files = self.create_data_dict(
            self.data_dir, 'train', X_train_subdir="/images", Y_train_subdir="/labels")
        train_files = self.shuffle_data(train_files, 'train')
        if data['val_from_train']:
            data_files['x_train'] = train_files['x_train'][data['val_limit']:data['train_limit']]
            data_files['y_train'] = train_files['y_train'][data['val_limit']:data['train_limit']]
            data_files['x_val'] = train_files['x_train'][:data['val_limit']]
            data_files['y_val'] = train_files['y_train'][:data['val_limit']]
        else:
            val_files = self.create_data_dict(
                self.data_dir, 'val', X_train_subdir="/images", Y_train_subdir="/labels")
            val_files = self.shuffle_data(val_files, 'val')
            data_files['x_train'] = train_files['x_train'][:data['train_limit']]
            data_files['y_train'] = train_files['y_train'][:data['train_limit']]
            data_files['x_val'] = val_files['x_val'][:data['val_limit']]
            data_files['y_val'] = val_files['y_val'][:data['val_limit']]
        for key in data_files.keys():
            print(key, len(data_files[key]))
        obj2h5(data_files, self.h5_dataset)
        return data_files

    def prepare_viz_data(self):
        data = {}
        data['colormap'] = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        train_viz = self.prepare_batch("train", 0, 8, False, False, False)
        val_viz = self.prepare_batch("val", 0, 8, False, False, False)
        train_prep = self.prepare_batch("train", 0, 8)
        val_prep = self.prepare_batch("val", 0, 8)
        data["x_train_viz"] = train_viz["x_train"]
        data["y_train_viz"] = train_viz["y_train"]
        data["x_val_viz"] = val_viz["x_val"]
        data["y_val_viz"] = val_viz["y_val"]
        data["x_train"] = train_prep["x_train"]
        data["x_val"] = val_prep["x_val"]
        obj2h5(data, self.h5_viz)

    def prepare_history(self):
        data = {}
        data['train_iou'] = []
        data['val_iou'] = []
        data['epoch'] = []
        obj2h5(data, self.h5_history)
    
    def prepare_iou_data(self):
        train_iou = self.prepare_batch('train', 0, self.val_samples, weighted_label=False)
        val_iou = self.prepare_batch(
            'val', 0, self.val_samples, weighted_label=False)
        data = {}
        data['x_train'] = train_iou['x_train']
        data['y_train'] = train_iou['y_train']
        data['x_val'] = val_iou['x_val']
        data['y_val'] = val_iou['y_val']
        obj2h5(data, self.h5_iou)

    def prepare_batch(self, state, start=None, end=None, normalize=True, hot_label=True, weighted_label=True):
        data = {}
        data["x_"+state], data["y_"+state] = self.load_image_and_seglabels(
            input_files=self.data_files["x_"+state][start:end],
            label_files=self.data_files["y_"+state][start:end],
            colormap=idcolormap,
            shape=self.shape,
            n_channels=self.n_channels,
            label_chanel_axis=self.label_chanel_axis,
            normalize=normalize,
            hot_label=hot_label)
        return data

    def create_file_lists(self, inputs_dir, labels_dir):
        label_files = glob.glob(os.path.join(labels_dir, "*.png"))
        file_ids = [os.path.basename(f).replace(
            "_L.png", ".png") for f in label_files]
        input_files = [os.path.join(inputs_dir, file_id[:-3]+'jpg')
                       for file_id in file_ids]
        label_files = [np.string_(f) for f in label_files]
        input_files = [np.string_(f) for f in input_files]
        input_files.sort()
        label_files.sort()
        return input_files, label_files

    def create_data_dict(self, datadir, state, X_train_subdir="/images", Y_train_subdir="/labels"):
        data = {}
        data["x_"+state], data["y_"+state] = self.create_file_lists(
            inputs_dir=os.path.join(datadir, state+X_train_subdir),
            labels_dir=os.path.join(datadir, state+Y_train_subdir))
        return data

    def load_image_and_seglabels(self, input_files, label_files, colormap, shape=(32, 32), n_channels=3, label_chanel_axis=False, normalize=True, hot_label=True, weighted_label=True):
        assert n_channels in {
            1, 3}, "Incorrect value for n_channels. Must be 1 or 3. Got {}".format(n_channels)

        width, height = shape
        n_samples = len(label_files)

        X = np.zeros([n_samples, height, width, n_channels], dtype=np.float32)
        if label_chanel_axis:
            Y = np.zeros([n_samples, height, width, 1], dtype=np.uint8)
        elif hot_label:
            Y = np.zeros([n_samples, height, width, self.n_classes], dtype=np.float32)
        else:
            Y = np.zeros([n_samples, height, width], dtype=np.uint8)

        for i in range(n_samples):
            img_file = input_files[i]
            label_file = label_files[i]
            img = PIL.Image.open(img_file).resize(
                shape, resample=PIL.Image.CUBIC)
            label_img = PIL.Image.open(label_file).resize(
                shape, resample=PIL.Image.NEAREST)
            img = np.asarray(img, dtype=np.float32)
            if normalize:
                img = np.divide(img, 255.0)
            label_img = np.asarray(label_img, dtype=np.float32)
            if colormap is not None:
                label_img = self.rgb2seglabel(
                    label_img, colormap=colormap, channels_axis=label_chanel_axis)
            if hot_label:
                label_img = tf.keras.utils.to_categorical(
                    y=label_img, num_classes=self.n_classes, dtype='float32')
            
            if weighted_label and hot_label:
                for j in range(self.n_classes):
                    label_img[label_img[:, :, j] == 1] *= self.weights['weights'][j]

            X[i] = img
            Y[i] = label_img

        return X, Y

    def rgb2seglabel(self, img, colormap, channels_axis=False):
        height, width, _ = img.shape
        if channels_axis:
            label = np.zeros([height, width, 1], dtype=np.float32)
        else:
            label = np.zeros([height, width], dtype=np.float32)
        for id in range(len(colormap)):
            label[np.all(img == np.array(colormap[id]), axis=2)] = id
        return label

    def calculate_class_weights(self, data):
        labels = self.data_files['y_train']
        random.shuffle(labels)
        counts = [0 for i in range(data['n_classes'])]
        total = 0
        weights = [0 for i in range(data['n_classes'])]
        for label_file in labels:
            label_img = PIL.Image.open(label_file).resize(
                (data['height'], data['width']), resample=PIL.Image.NEAREST)
            label_img = np.asarray(label_img, dtype=np.uint8)
            label_img = self.rgb2seglabel(label_img, idcolormap)
            ids, c = np.unique(label_img, return_counts=True)
            if total + sum(c) < 0:
                break
            total += sum(c)
            for i in range(len(ids)):
                if counts[ids[i]] + c[i] < 0:
                    continue
                counts[ids[i]] += c[i]
                weights[ids[i]] = (1 / counts[ids[i]])*(total)/2.0
            
            sys.stdout.write('\rCalculating Class Weights: '+str(weights))
            sys.stdout.flush()
        print('\n')
        d = {}
        d['weights'] = weights
        obj2h5(d, self.h5_weights)

    def shuffle_data(self, data, state):
        a, b = data['x_'+state], data['y_'+state]
        c = list(zip(a, b))
        random.shuffle(c)
        a, b = zip(*c)
        data['x_'+state], data['y_'+state] = a, b
        return data


def obj2h5(data, h5_file):
    with h5py.File(h5_file, 'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=np.array(value))
        f.close()


def h52obj(file):
    data = {}
    with h5py.File(file, 'r') as f:
        for key, value in f.items():
            data[key] = np.array(value)
        f.close()
    return data
