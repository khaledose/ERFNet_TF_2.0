import numpy as np
import matplotlib.pyplot as plt


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
