from __future__ import print_function, division, unicode_literals
import glob
import os
import PIL.Image
import numpy as np
import h5py

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
    def __init__(self, data_dir, width, height, limit, val_limit, n_classes):
        self.data_dir = data_dir  # "/content/ERFNet_TF_2.0/km10k/"
        self.h5_file = self.data_dir+"data.h5"
        self.width, self.height = width, height
        self.shape = [self.width, self.height]
        self.n_channels = 3
        self.label_chanel_axis = False
        self.data = {}
        if os.path.isfile(self.h5_file):
            print("Data H5 "+self.h5_file+" is found")
            self.data = self.prepare_data(data_file=self.h5_file, n_classes=n_classes, valid_from_train=True,
                                          n_valid=val_limit, max_data=None)
            return
        print("CREATING DATA")
        print("- Getting list of files")
        self.file_data = self.create_data_dict(
            self.data_dir, X_train_subdir="images", Y_train_subdir="labels", limit=limit)
        n_samples = len(self.file_data["x_train"])

        est_size = n_samples*width*height*(3+1)/(1024*1000)
        print("- Estimated data size is {} MB (+ overhead)".format(est_size))

        print("- Loading image files and converting to arrays")

        self.data["x_train"], self.data["y_train"] = self.load_image_and_seglabels(
            input_files=self.file_data["x_train"],
            label_files=self.file_data["y_train"],
            colormap=idcolormap,
            shape=self.shape,
            n_channels=self.n_channels,
            label_chanel_axis=self.label_chanel_axis)

        print("- H5pying the data to:", self. h5_file)
        obj2h5(self.data, self.h5_file)
        self.data = self.prepare_data(data_file=self.h5_file, n_classes=n_classes, valid_from_train=True,
                                      n_valid=val_limit, max_data=None)
        print("- DONE!")

    def create_file_lists(self, inputs_dir, labels_dir, limit):
        label_files = glob.glob(os.path.join(labels_dir, "*.png"))[:limit]
        file_ids = [os.path.basename(f).replace(
            "_L.png", ".png") for f in label_files[:limit]]
        input_files = [os.path.join(inputs_dir, file_id[:-3]+'jpg')
                       for file_id in file_ids]
        print(len(input_files), len(label_files))
        return input_files, label_files

    def create_data_dict(self, datadir, X_train_subdir="train_inputs", Y_train_subdir="train_labels", limit=1000):
        data = {}
        data["x_train"], data["y_train"] = self.create_file_lists(
            inputs_dir=os.path.join(datadir, X_train_subdir),
            labels_dir=os.path.join(datadir, Y_train_subdir), limit=1000)
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

    def prepare_data(self, data_file, n_classes, valid_from_train=False, n_valid=1024, max_data=None, verbose=True):
        print("Preparing Data Dictionary")
        ds = h52obj(data_file)
        data = {}
        if valid_from_train:
            data["x_val"] = ds["x_train"][:n_valid]
            data["y_val"] = ds["y_train"][:n_valid]
            data["x_train"] = ds["x_train"][n_valid:]
            data["y_train"] = ds["y_train"][n_valid:]

        if max_data:
            data["x_train"] = ds["x_train"][:max_data]
            data["y_train"] = ds["y_train"][:max_data]

        data["x_train_viz"] = ds["x_train"][:25]
        data["y_train_viz"] = ds["y_train"][:25]

        data["id2label"] = id2label
        data["label2id"] = label2id
        data['colormap'] = [(0, 0, 0), (255, 0, 0), (0, 255, 0),
                            (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
        data['weights'] = self.calculate_class_weights(
            data["y_train"], n_classes)
        data['n_classes'] = n_classes

        if verbose:
            print("DATA SHAPES")
            print("- X_valid: ", (data["x_val"]).shape)
            print("- Y_valid: ", (data["y_val"]).shape)
            print("- X_train: ", (data["x_train"]).shape)
            print("- Y_train: ", (data["y_train"]).shape)
            if "X_test" in data:
                print("- X_test: ", (data["x_test"]).shape)
                print("- Y_test: ", (data["y_test"]).shape)

        return data

    def calculate_class_weights(self, Y, n_classes, method="paszke", c=1.02):
        ids, counts = np.unique(Y, return_counts=True)
        n_pixels = Y.size
        p_class = np.zeros(n_classes)
        p_class[ids] = counts/n_pixels
        if method == "paszke":
            weights = 1/np.log(c+p_class)
        elif method == "eigen":
            assert False, "TODO: Implement eigen method"
        elif method in {"eigen2", "logeigen2"}:
            epsilon = 1e-8
            median = np.median(p_class)
            weights = median/(p_class+epsilon)
            if method == "logeigen2":
                weights = np.log(weights+1)
        else:
            assert False, "Incorrect choice for method"

        return weights

    def shuffle_train_data(self, data):
        n_samples = len(data["y_train"])
        permutation = list(np.random.permutation(n_samples))
        data["x_train"] = data["x_train"][permutation]
        data["y_train"] = data["y_train"][permutation]
        return data


def obj2h5(data, h5_file):
    with h5py.File(h5_file, 'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)


def h52obj(file):
    data = {}
    with h5py.File(file, 'r') as f:
        for key, value in f.items():
            data[key] = value[:]
    return data
