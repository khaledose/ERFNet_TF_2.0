from __future__ import print_function, division, unicode_literals
import glob
import os
import numpy as np
import h5py
import cv2

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


def create_file_lists(inputs_dir, labels_dir):
    """ Given the paths to the directories containing the input and label
        images, it creates a list of the full filepaths for those images,
        with the same ordering, so the same index in each list represents
        the corresponding input/label pair.

        Returns 2-tuple of two lists: (input_files, label_files)
    """
    label_files = glob.glob(os.path.join(labels_dir, "*.png"))
    file_ids = [os.path.basename(f).replace("_L.png", ".png")
                for f in label_files]
    input_files = [os.path.join(inputs_dir, file_id[:-3]+'jpg')
                   for file_id in file_ids]
    return input_files, label_files


def create_data_dict(datadir, X_train_subdir="train_inputs", Y_train_subdir="train_labels"):
    data = {}
    data["X_train"], data["Y_train"] = create_file_lists(
        inputs_dir=os.path.join(datadir, X_train_subdir),
        labels_dir=os.path.join(datadir, Y_train_subdir))
    return data


def calculate_class_weights(Y, n_classes, method="paszke", c=1.02):
    """ Given the training data labels Calculates the class weights.

    Args:
        Y:      (numpy array) The training labels as class id integers.
                The shape does not matter, as long as each element represents
                a class id (ie, NOT one-hot-vectors).
        n_classes: (int) Number of possible classes.
        method: (str) The type of class weighting to use.

                - "paszke" = use the method from from Paszke et al 2016
                            `1/ln(c + class_probability)`
                - "eigen"  = use the method from Eigen & Fergus 2014.
                             `median_freq/class_freq`
                             where `class_freq` is based only on images that
                             actually contain that class.
                - "eigen2" = Similar to `eigen`, except that class_freq is
                             based on the frequency of the class in the
                             entire dataset, not just images where it occurs.
                -"logeigen2" = takes the log of "eigen2" method, so that
                            incredibly rare classes do not completely overpower
                            other values.
        c:      (float) Coefficient to use, when using paszke method.

    Returns:
        weights:    (numpy array) Array of shape [n_classes] assigning a
                    weight value to each class.

    References:
        Eigen & Fergus 2014: https://arxiv.org/abs/1411.4734
        Paszke et al 2016: https://arxiv.org/abs/1606.02147
    """
    # CLASS PROBABILITIES - based on empirical observation of data
    ids, counts = np.unique(Y, return_counts=True)
    n_pixels = Y.size
    p_class = np.zeros(n_classes)
    p_class[ids] = counts/n_pixels

    # CLASS WEIGHTS
    if method == "paszke":
        weights = 1/np.log(c+p_class)
    elif method == "eigen":
        assert False, "TODO: Implement eigen method"
        # TODO: Implement eigen method
        # where class_freq is the number of pixels of class c divided by
        # the total number of pixels in images where c is actually present,
        # and median freq is the median of these frequencies.
    elif method in {"eigen2", "logeigen2"}:
        epsilon = 1e-8  # to prevent division by 0
        median = np.median(p_class)
        weights = median/(p_class+epsilon)
        if method == "logeigen2":
            weights = np.log(weights+1)
    else:
        assert False, "Incorrect choice for method"

    return weights


def load_image_and_seglabels(input_files, label_files, colormap, shape=(32, 32), n_channels=3, label_chanel_axis=False):
    """ Given a list of input image file paths and corresponding segmentation
        label image files (with different RGB values representing different
        classes), and a colormap list, it:

        - loads up the images
        - resizes them to a desired shape
        - converts segmentation labels to single color channel image with
          integer value of pixel representing the class id.

    Args:
        input_files:        (list of str) file paths for input images
        label_files:        (list of str) file paths for label images
        colormap:           (list or None) A list where each index represents the
                            color value for the corresponding class id.
                            Eg: for RGB labels, to map class_0 to black and
                            class_1 to red:
                                [(0,0,0), (255,0,0)]
                            Set to None if images are already encoded as
                            greyscale where the integer value represents the
                            class id.
        shape:              (2-tuple of ints) (width,height) to reshape images
        n_channels:         (int) Number of chanels for input images
        label_chanel_axis:  (bool)(default=False) Use color chanel axis for
                            array of label images?
    """
    # Dummy proofing
    assert n_channels in {
        1, 3}, "Incorrect value for n_channels. Must be 1 or 3. Got {}".format(n_channels)

    # Image dimensions
    width, height = shape
    n_samples = len(label_files)

    # Initialize input and label batch arrays
    X = np.zeros([n_samples, height, width, n_channels], dtype=np.uint8)
    if label_chanel_axis:
        Y = np.zeros([n_samples, height, width, 1], dtype=np.uint8)
    else:
        Y = np.zeros([n_samples, height, width], dtype=np.uint8)

    for i in range(n_samples):
        # Get filenames of input and label
        img_file = input_files[i]
        label_file = label_files[i]

        # Resize input and label images
        img = cv2.imread(img_file)
        img = cv2.resize(img, shape, interpolation=cv2.INTER_CUBIC)
        label_img = cv2.imread(label_file)
        label_img = cv2.resize(
            label_img, shape, interpolation=cv2.INTER_NEAREST)

        # Convert label image from RGB to single value int class labels
        if colormap is not None:
            label_img = cv2.cvtColor(label_img, cv2.COLOR_BGR2GRAY)

        # Add processed images to batch arrays
        X[i] = img
        Y[i] = label_img

    return X, Y


def prepare_data(data_file, valid_from_train=False, n_valid=1024, max_data=None, verbose=True):
    # data = pickle2obj(data_file)
    ds = h52obj(data_file)
    data = {}
    # Create validation from train data
    if valid_from_train:
        data["X_valid"] = ds["X_train"][:n_valid]
        data["Y_valid"] = ds["Y_train"][:n_valid]
        data["X_train"] = ds["X_train"][n_valid:]
        data["Y_train"] = ds["Y_train"][n_valid:]

    if max_data:
        data["X_train"] = ds["X_train"][:max_data]
        data["Y_train"] = ds["Y_train"][:max_data]

    # Visualization data
    n_viz = 25
    data["X_train_viz"] = ds["X_train"][:25]
    data["Y_train_viz"] = ds["Y_train"][:25]

    data["id2label"] = id2label
    data["label2id"] = label2id
    data['colormap'] = [(0, 0, 0), (255, 0, 0), (0, 255, 0),
                        (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]

    if verbose:
        # Print information about data
        print("DATA SHAPES")
        print("- X_valid: ", (data["X_valid"]).shape)
        print("- Y_valid: ", (data["Y_valid"]).shape)
        print("- X_train: ", (data["X_train"]).shape)
        print("- Y_train: ", (data["Y_train"]).shape)
        if "X_test" in data:
            print("- X_test: ", (data["X_test"]).shape)
            print("- Y_test: ", (data["Y_test"]).shape)

    return data


def obj2h5(data, h5_file):
    with h5py.File(h5_file, 'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)


def h52obj(file):
    data = {}
    with h5py.File(file, 'r') as f:
        for key, value in f.items():
            data[key] = np.array(value)
    return data


if __name__ == '__main__':
    # SETTINGS
    data_dir = "/content/erfnet_segmentation/km10k/"
    h5_file = "data.h5"
    shape = (480, 640)
    width, height = shape
    n_channels = 3
    label_chanel_axis = False  # Create chanels axis for label images?

    print("CREATING DATA")
    print("- Getting list of files")
    file_data = create_data_dict(
        data_dir, X_train_subdir="images", Y_train_subdir="labels")
    n_samples = len(file_data["X_train"])

    est_size = n_samples*width*height*(3+1)/(1024*1000)
    print("- Estimated data size is {} MB (+ overhead)".format(est_size))

    print("- Loading image files and converting to arrays")
    data = {}
    data["X_train"], data["Y_train"] = load_image_and_seglabels(
        input_files=file_data["X_train"],
        label_files=file_data["Y_train"],
        colormap=idcolormap,
        shape=shape,
        n_channels=n_channels,
        label_chanel_axis=label_chanel_axis)

    print("- Saving data as HDF5 to: ", h5_file)
    obj2h5(data, h5_file)
    print("- DONE!")
