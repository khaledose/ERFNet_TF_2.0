import numpy as np
import matplotlib.pyplot as plt
import PIL
import os


def get_mask(mask, colormap):
    mask = np.array(colormap)[mask].astype(np.uint8)
    mask = mask[:, :, ::-1]
    return mask


def get_class_masks(y_true, y_pred, n_classes, colormap):
    gt = {}
    pred = {}
    for i in range(n_classes):
        im1 = np.all(y_true == colormap[i], axis=-1)
        im2 = np.all(y_pred == colormap[i], axis=-1)
        if(len(np.unique(im1)) < 2 and len(np.unique(im2)) < 2):
            continue
        im11 = np.zeros((480, 640), dtype=np.int16)
        im22 = np.zeros((480, 640), dtype=np.int16)
        im11[im1] = 255
        im22[im2] = 255
        gt[i] = im11
        pred[i] = im22
        im1 = None
        im2 = None
        im11 = None
        im22 = None
    return gt, pred


def draw_training_curve(train, valid, saveto, title, ylab, legend_pos):
    fig, ax = plt.subplots(figsize=(6, 4))
    fig.suptitle(title, fontsize=15)
    ax.plot(train, color="#FF4F40",  label="train")
    ax.plot(valid, color="#307EC7",  label="valid")
    ax.set_xlabel("epoch")
    ax.set_ylabel(ylab)

    ax.grid(True)
    plt.minorticks_on()
    plt.grid(b=True, which='major', color='#888888', linestyle='-')
    plt.grid(b=True, which='minor', color='#AAAAAA', linestyle='-', alpha=0.2)

    ax.legend(loc=legend_pos, title="", frameon=False,  fontsize=8)
    fig.savefig(saveto)
    plt.close()


def viz_segmentation_pairs(X, Y, Y2=None, colormap=None, gridshape=(2, 8), saveto=None):
    assert (X.ndim == 3) or (
        X.ndim == 4 and X.shape[-1] in {1, 3}), "X is wrong dimensions"
    assert (Y.ndim == 3), "Y is wrong dimensions"
    assert (Y2 is None) or (Y2.ndim == 3), "Y2 is wrong dimensions"

    rows, cols = gridshape
    assert rows > 0 and cols > 0, "rows and cols must be positive integers"
    n_cells = (rows*cols)
    X = X[:n_cells]
    n_samples = X.shape[0]

    if (X.ndim == 3):
        X = np.expand_dims(X, axis=3)
    output = []
    for i in range(min(n_cells, n_samples)):
        x = X[i]
        y = Y[i]
        if x.shape[-1] == 1:
            x = np.repeat(x, 3, axis=2)

        y = np.array(colormap)[y].astype(np.uint8)
        if Y2 is None:
            output.append(np.concatenate([x, y], axis=0))
        else:
            y2 = Y2[i]
            y2 = np.array(colormap)[y2].astype(np.uint8)
            output.append(np.concatenate([x, y, y2], axis=0))

    output = np.array(output, dtype=np.uint8)
    output = batch2grid(output, rows=rows, cols=cols)
    output = PIL.Image.fromarray(output.squeeze())

    if saveto is not None:
        pardir = os.path.dirname(saveto)
        if pardir.strip() != "":
            if not os.path.exists(pardir):
                os.makedirs(pardir)
        output.save(saveto, "JPEG")

    return output


def batch2grid(imgs, rows, cols):
    assert rows > 0 and cols > 0, "rows and cols must be positive integers"

    # Prepare dimensions of the grid
    n_cells = (rows*cols)
    imgs = imgs[:n_cells]  # Only use the number of images needed to fill grid

    # Image dimensions
    n_dims = imgs.ndim
    assert n_dims == 3 or n_dims == 4, "Incorrect # of dimensions for input array"

    # Deal with images that have no color channel
    if n_dims == 3:
        imgs = np.expand_dims(imgs, axis=3)

    # get dimensions of image
    n_samples, img_height, img_width, n_channels = imgs.shape

    # Handle case where there is not enough images in batch to fill grid
    if n_cells > n_samples:
        n_gap = n_cells - n_samples
        imgs = np.pad(imgs, pad_width=[
                      (0, n_gap), (0, 0), (0, 0), (0, 0)], mode="constant", constant_values=0)

    # Reshape into grid
    grid = imgs.reshape(rows, cols, img_height, img_width,
                        n_channels).swapaxes(1, 2)
    grid = grid.reshape(rows*img_height, cols*img_width, n_channels)

    # If input was flat images with no color channels, then flatten the output
    if n_dims == 3:
        # axis 2 because batch dim has been removed
        grid = grid.squeeze(axis=2)

    return grid
