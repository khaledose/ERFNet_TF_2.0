from __future__ import print_function, division, unicode_literals
import glob
import os
import PIL.Image
import numpy as np
import pickle
import h5py
__author__ = "Ronny Restrepo"
__copyright__ = "Copyright 2017, Ronny Restrepo"
__credits__ = ["Ronny Restrepo"]
__license__ = "Apache License"
__version__ = "2.0"


label_colormap = {
    "Void":(0,0,0,255),
    "Crosswalk":(255,0,0,255),
    "Double White":(0,255,0,255),
    "Double Yellow":(0,0,255,255),
    "Road Curb":(255,255,0,255),
    "Single White":(255,0,255,255),
    "Single Yellow":(0,255,255,255)
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

label2id = {label:id for id, label in enumerate(id2label)}
idcolormap = [label_colormap[label] for label in id2label]

# Check nothing stupid happened with mappings
assert set(label_colormap) == set(id2label) == set(label2id.keys()), "Something is wrong with the id label maps"


# ==============================================================================
#                                                                 MAYBE_MAKE_DIR
# ==============================================================================
def maybe_make_dir(path):
    """ Checks if a directory path exists on the system, if it does not, then
        it creates that directory (and any parent directories needed to
        create that directory)
    """
    if not os.path.exists(path):
        os.makedirs(path)

# ==============================================================================
#                                                                     GET_PARDIR
# ==============================================================================
def get_pardir(file):
    """ Given a file path, it returns the parent directory of that file. """
    return os.path.dirname(file)


# ==============================================================================
#                                                              MAYBE_MAKE_PARDIR
# ==============================================================================
def maybe_make_pardir(file):
    """ Takes a path to a file, and creates the necessary directory structure
        on the system to ensure that the parent directory exists (if it does
        not already exist)
    """
    pardir = os.path.dirname(file)
    if pardir.strip() != "": # ensure pardir is not an empty string
        if not os.path.exists(pardir):
            os.makedirs(pardir)


# ==============================================================================
#                                                                       FILE2STR
# ==============================================================================
def file2str(file):
    """ Takes a file path and returns the contents of that file as a string."""
    with open(file, "r") as textFile:
        return textFile.read()

# ==============================================================================
#                                                                       STR2FILE
# ==============================================================================
def str2file(s, file, mode="w"):
    """ Writes a string to a file"""
    # Ensure parent directory and necesary file structure exists
    pardir = os.path.dirname(file)
    if pardir.strip() != "": # ensure pardir is not an empty string
        if not os.path.exists(pardir):
            os.makedirs(pardir)

    with open(file, mode=mode) as textFile:
        textFile.write(s)


# ==============================================================================
#                                                                     OBJ2PICKLE
# ==============================================================================
def obj2pickle(obj, file, protocol=4):
    """ Saves an object as a binary pickle file to the desired file path. """
    # Ensure parent directory and necesary file structure exists
    pardir = os.path.dirname(file)
    if pardir.strip() != "": # ensure pardir is not an empty string
        if not os.path.exists(pardir):
            os.makedirs(pardir)

    with open(file, mode="wb") as fileObj:
        pickle.dump(obj, fileObj, protocol=protocol)


# ==============================================================================
#                                                                     PICKLE2OBJ
# ==============================================================================
def pickle2obj(file):
    """ Loads the contents of a pickle as a python object. """
    with open(file, mode = "rb") as fileObj:
        obj = pickle.load(fileObj)
    return obj


# ==============================================================================
#                                                              CREATE_FILE_LISTS
# ==============================================================================
def create_file_lists(inputs_dir, labels_dir):
    """ Given the paths to the directories containing the input and label
        images, it creates a list of the full filepaths for those images,
        with the same ordering, so the same index in each list represents
        the corresponding input/label pair.

        Returns 2-tuple of two lists: (input_files, label_files)
    """
    # Create (synchronized) lists of full file paths to input and label images
    label_files = glob.glob(os.path.join(labels_dir, "*.png"))
    file_ids = [os.path.basename(f).replace("_L.png", ".png") for f in label_files]
    input_files = [os.path.join(inputs_dir, file_id[:-3]+'jpg') for file_id in file_ids]
    return input_files, label_files


# ==============================================================================
#                                                               CREATE_DATA_DICT
# ==============================================================================
def create_data_dict(datadir, X_train_subdir="train_inputs", Y_train_subdir="train_labels"):
    data = {}
    data["x_train"], data["y_train"] = create_file_lists(
        inputs_dir=os.path.join(datadir, X_train_subdir),
        labels_dir=os.path.join(datadir, Y_train_subdir))
    return data


# ==============================================================================
#                                                              PIXELS_WITH_VALUE
# ==============================================================================
def pixels_with_value(img, val):
    """ Given an image as a numpy array, and a value representing the
        pixel values, eg [128,255,190] in an RGB image, then it returns
        a 2D boolean array with a True for every pixel position that has
        that value.
    """
    return np.all(img==np.array(val), axis=2)


# ==============================================================================
#                                                                   RGB2SEGLABEL
# ==============================================================================
def rgb2seglabel(img, colormap, channels_axis=False):
    """ Given an RGB image stored as a numpy array, and a colormap that
        maps from label id to an RGB color for that label, it returns a
        new numpy array with color chanel size of 1 where the pixel
        intensity values represent the class label id.

    Args:
        img:            (np array)
        colormap:       (list) list of pixel values for each class label id
        channels_axis:  (bool)(default=False) Should it return an array with a
                        third (color channels) axis of size 1?
    """
    height, width, _ = img.shape
    if channels_axis:
        label = np.zeros([height, width,1], dtype=np.uint8)
    else:
        label = np.zeros([height, width], dtype=np.uint8)
    for id in range(len(colormap)):
        label[np.all(img==np.array(idcolormap[id]), axis=2)] = id
    return label


# ==============================================================================
#                                                       LOAD_IMAGE_AND_SEGLABELS
# ==============================================================================
def load_image_and_seglabels(input_files, label_files, colormap, shape=(32,32), n_channels=3, label_chanel_axis=False):
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
    assert n_channels in {1,3}, "Incorrect value for n_channels. Must be 1 or 3. Got {}".format(n_channels)

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
        img = PIL.Image.open(img_file).resize(shape, resample=PIL.Image.CUBIC)
        label_img = PIL.Image.open(label_file).resize(shape, resample=PIL.Image.NEAREST)

        # Convert back to numpy arrays
        img = np.asarray(img, dtype=np.uint8)
        label_img = np.asarray(label_img, dtype=np.uint8)

        # Convert label image from RGB to single value int class labels
        if colormap is not None:
            label_img = rgb2seglabel(label_img, colormap=colormap, channels_axis=label_chanel_axis)

        # Add processed images to batch arrays
        X[i] = img
        Y[i] = label_img

    return X, Y


# ==============================================================================
#                                                                   PREPARE_DATA
# ==============================================================================

def prepare_data(data_file, valid_from_train=False, n_valid=1024, max_data=None, verbose=True):
    # data = pickle2obj(data_file)
    ds = h52obj(data_file)
    data = {}
    # Create validation from train data
    if valid_from_train:
        data["x_val"] = ds["x_train"][:n_valid]
        data["y_val"] = ds["y_train"][:n_valid]
        data["x_train"] = ds["x_train"][n_valid:]
        data["y_train"] = ds["y_train"][n_valid:]

    if max_data:
        data["x_train"] = ds["x_train"][:max_data]
        data["y_train"] = ds["y_train"][:max_data]

    # Visualization data
    data["x_train_viz"] = ds["x_train"][:25]
    data["y_train_viz"] = ds["y_train"][:25]

    data["id2label"] = id2label
    data["label2id"] = label2id
    data['colormap'] = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    data['weights'] = [0.21706227, 3.87922712, 9.64396687,
                       6.6800881,  3.99014264, 2.35461766, 7.59079911]

    if verbose:
        # Print information about data
        print("DATA SHAPES")
        print("- X_valid: ", (data["x_val"]).shape)
        print("- Y_valid: ", (data["y_val"]).shape)
        print("- X_train: ", (data["x_train"]).shape)
        print("- Y_train: ", (data["y_train"]).shape)
        if "X_test" in data:
            print("- X_test: ", (data["x_test"]).shape)
            print("- Y_test: ", (data["y_test"]).shape)

    return data


# ==============================================================================
#                                                        CALCULATE_CLASS_WEIGHTS
# ==============================================================================
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
        epsilon = 1e-8 # to prevent division by 0
        median = np.median(p_class)
        weights = median/(p_class+epsilon)
        if method == "logeigen2":
            weights = np.log(weights+1)
    else:
        assert False, "Incorrect choice for method"

    return weights

def obj2h5(data, h5_file):
    with h5py.File(h5_file, 'w') as f:
        for key, value in data.items():
            f.create_dataset(key, data=value)

def h52obj(file):
    data = {}
    with h5py.File(file, 'r') as f:
        for key, value in f.items():
            print(value[:].shape)
            data[key] = value[:]
    return data

def shuffle_train_data(data):
    n_samples = len(data["y_train"])
    permutation = list(np.random.permutation(n_samples))
    data["x_train"] = data["x_train"][permutation]
    data["y_train"] = data["y_train"][permutation]
    return data

if __name__ == '__main__':
    # SETTINGS
    data_dir = "/content/erfnet_segmentation/km10k/"
    h5_file = "data.h5"
    shape = [480, 640]
    width, height = shape
    n_channels = 3
    label_chanel_axis=False # Create chanels axis for label images?

    print("CREATING DATA")
    print("- Getting list of files")
    file_data = create_data_dict(data_dir, X_train_subdir="images", Y_train_subdir="labels")
    n_samples = len(file_data["x_train"])

    est_size = n_samples*width*height*(3+1)/(1024*1000)
    print("- Estimated data size is {} MB (+ overhead)".format(est_size))

    print("- Loading image files and converting to arrays")
    data = {}
    data["x_train"], data["y_train"] = load_image_and_seglabels(
        input_files=file_data["x_train"],
        label_files=file_data["y_train"],
        colormap=idcolormap,
        shape=shape,
        n_channels=n_channels,
        label_chanel_axis=label_chanel_axis)

    print("- H5pying the data to:", h5_file)
    obj2h5(data, h5_file)
    print("- DONE!")
