import matplotlib.pyplot as plt
import numpy as np
import PIL
import os


def get_mask(mask, colormap):
    mask = np.array(colormap)[mask].astype(np.uint8)
    mask = mask[:, :, ::-1]
    return mask


def draw_samples(samples, colormap, epoch, state, saveto):
    viz_img_template = os.path.join(
        saveto,
        "samples",
        "{}",
        "epoch_{: 07d}.jpg",
    )
    for i in range(8):
        viz_segmentation_pairs(
            X=samples[0],
            Y=samples[1],
            Y2=samples[2],
            colormap=colormap,
            gridshape=(2, 4),
            saveto=viz_img_template.format(state, epoch)
        )


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

    n_cells = (rows*cols)
    imgs = imgs[:n_cells]

    n_dims = imgs.ndim
    assert n_dims == 3 or n_dims == 4, "Incorrect # of dimensions for input array"

    if n_dims == 3:
        imgs = np.expand_dims(imgs, axis=3)

    n_samples, img_height, img_width, n_channels = imgs.shape

    if n_cells > n_samples:
        n_gap = n_cells - n_samples
        imgs = np.pad(imgs, pad_width=[
                      (0, n_gap), (0, 0), (0, 0), (0, 0)], mode="constant", constant_values=0)

    grid = imgs.reshape(rows, cols, img_height, img_width,
                        n_channels).swapaxes(1, 2)
    grid = grid.reshape(rows*img_height, cols*img_width, n_channels)

    if n_dims == 3:
        grid = grid.squeeze(axis=2)

    return grid
