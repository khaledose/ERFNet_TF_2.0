from __future__ import print_function, division, unicode_literals
import glob
import os
import PIL.Image
import numpy as np
import h5py
import random
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
    def __init__(self, data_dir, width, height, limit, val_limit, n_classes, train_method, val_from_train=False):
        self.data_dir = data_dir
        self.h5_file = self.data_dir+"data.h5"
        self.h5_stuff = self.data_dir+"stuff.h5"
        self.width, self.height = width, height
        self.shape = [self.width, self.height]
        self.n_channels = 3
        self.label_chanel_axis = False
        self.data_files = self.prepare_data_files(
            val_from_train, limit, val_limit)
        self.data = {}
        self.stuff = {}
        print('Preparing Data')
        if train_method == 0:
            if os.path.isfile(self.h5_file):
                self.data = h52obj(self.h5_file, 0, limit)
            else:
                self.data = self.prepare_batch('train', end=limit)
                obj2h5(self.data, self.h5_file)

        if os.path.isfile(self.h5_stuff):
            self.stuff = h52obj(self.h5_stuff)
        else:
            val_data = self.prepare_batch('val')
            train_data = self.prepare_batch('train', end=val_limit)
            self.stuff = self.prepare_data(
                train_data, val_data, n_classes=n_classes, n_valid=val_limit)
            obj2h5(self.stuff, self.h5_stuff)
        print('Done')

    def prepare_data_files(self, val_from_train, limit, val_limit):
        data_files = {}
        train_files = self.create_data_dict(
            self.data_dir, 'train', X_train_subdir="/images", Y_train_subdir="/labels", end=limit)
        train_files = self.shuffle_train_data(train_files)
        if val_from_train:
            data_files['x_train'] = train_files['x_train'][val_limit:]
            data_files['y_train'] = train_files['y_train'][val_limit:]
            data_files['x_val'] = train_files['x_train'][:val_limit]
            data_files['y_val'] = train_files['y_train'][:val_limit]
        else:
            val_files = self.create_data_dict(
                self.data_dir, 'val', X_train_subdir="/images", Y_train_subdir="/labels", end=val_limit)
            data_files['x_train'] = train_files['x_train']
            data_files['y_train'] = train_files['y_train']
            data_files['x_val'] = val_files['x_val']
            data_files['y_val'] = val_files['y_val']
        return data_files

    def prepare_batch(self, state, start=0, end=None):
        data = {}
        data["x_"+state], data["y_"+state] = self.load_image_and_seglabels(
            input_files=self.data_files["x_"+state][start:end],
            label_files=self.data_files["y_"+state][start:end],
            colormap=idcolormap,
            shape=self.shape,
            n_channels=self.n_channels,
            label_chanel_axis=self.label_chanel_axis)
        return data

    def create_file_lists(self, inputs_dir, labels_dir, start=0, end=16):
        label_files = glob.glob(os.path.join(labels_dir, "*.png"))[start:end]
        file_ids = [os.path.basename(f).replace(
            "_L.png", ".png") for f in label_files]
        input_files = [os.path.join(inputs_dir, file_id[:-3]+'jpg')
                       for file_id in file_ids]
        input_files.sort()
        label_files.sort()
        return input_files, label_files

    def create_data_dict(self, datadir, state, X_train_subdir="/images", Y_train_subdir="/labels", start=0, end=16):
        data = {}
        data["x_"+state], data["y_"+state] = self.create_file_lists(
            inputs_dir=os.path.join(datadir, state+X_train_subdir),
            labels_dir=os.path.join(datadir, state+Y_train_subdir), start=start, end=end)
        return data

    def load_image_and_seglabels(self, input_files, label_files, colormap, shape=(32, 32), n_channels=3, label_chanel_axis=False):
        assert n_channels in {
            1, 3}, "Incorrect value for n_channels. Must be 1 or 3. Got {}".format(n_channels)

        width, height = shape
        n_samples = len(label_files)

        X = np.zeros([n_samples, height, width, n_channels], dtype=np.uint8)
        if label_chanel_axis:
            Y = np.zeros([n_samples, height, width, 1], dtype=np.uint8)
        else:
            Y = np.zeros([n_samples, height, width], dtype=np.uint8)

        for i in range(n_samples):
            img_file = input_files[i]
            label_file = label_files[i]
            img = PIL.Image.open(img_file).resize(
                shape, resample=PIL.Image.CUBIC)
            label_img = PIL.Image.open(label_file).resize(
                shape, resample=PIL.Image.NEAREST)
            img = np.asarray(img, dtype=np.uint8)
            label_img = np.asarray(label_img, dtype=np.uint8)

            if colormap is not None:
                label_img = self.rgb2seglabel(
                    label_img, colormap=colormap, channels_axis=label_chanel_axis)

            X[i] = img
            Y[i] = label_img

        return X, Y

    def rgb2seglabel(self, img, colormap, channels_axis=False):
        height, width, _ = img.shape
        if channels_axis:
            label = np.zeros([height, width, 1], dtype=np.uint8)
        else:
            label = np.zeros([height, width], dtype=np.uint8)
        for id in range(len(colormap)):
            label[np.all(img == np.array(idcolormap[id]), axis=2)] = id
        return label

    def prepare_data(self, train_data, val_data, n_classes, n_valid=1024):
        print("Preparing Data Dictionary")
        data = {}
        data["x_val"] = val_data["x_val"]
        data["y_val"] = val_data["y_val"]
        data["x_train"] = train_data["x_train"]
        data["y_train"] = train_data["y_train"]
        train_data = None
        val_data = None
        data["x_train_viz"] = data["x_train"][:8]
        data["y_train_viz"] = data["y_train"][:8]
        data["x_val_viz"] = data["x_val"][:8]
        data["y_val_viz"] = data["y_val"][:8]
        data['colormap'] = [(0, 0, 0), (255, 0, 0), (0, 255, 0),
                            (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        data['weights'] = [1.43153438, 46.09928655, 50.26547627, 48.96395612, 46.30125866, 41.71282607, 49.49144053]
        data['n_classes'] = [n_classes]
        return data

    def calculate_class_weights(self, Y, n_classes, c=1.02):
        ids, counts = np.unique(Y, return_counts=True)
        n_pixels = Y.size
        p_class = np.zeros(n_classes)
        p_class[ids] = counts/n_pixels
        weights = 1/np.log(c+p_class)
        return weights

    def shuffle_train_data(self, data):
        # n_samples = len(data["y_train"])
        # permutation = list(np.random.permutation(n_samples))
        # data["x_train"] = data["x_train"][permutation]
        # data["y_train"] = data["y_train"][permutation]
        a, b = data['x_train'], data['y_train']
        c = list(zip(a, b))
        random.shuffle(c)
        a, b = zip(*c)
        data['x_train'], data['y_train'] = a, b
        return data


def obj2h5(data, h5_file, mini=0, maxi=None):
    with h5py.File(h5_file, 'w') as f:
        for key, value in data.items():
            if maxi != None:
                f.create_dataset(key, data=value[mini:maxi])
            else:
                f.create_dataset(key, data=value[mini:])
        f.close()


def h52obj(file, mini=0, maxi=None):
    data = {}
    with h5py.File(file, 'r') as f:
        for key, value in f.items():
            if maxi != None:
                data[key] = value[mini:maxi]
            else:
                data[key] = value[:]
        f.close()
    return data
