import numpy as np
import tensorflow as tf
from data_processing import prepare_data, shuffle_train_data
from model_arch import ERFNet
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
import cv2


class IoUCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        iou_t = self.print_iou('x_train', 'y_train', 10)
        iou_v = self.print_iou('x_val', 'y_val', 10)
        print()
        print("Epoch "+str(epoch+1)+": Train IoU = " +
              str(iou_t), "Validation IoU of  = " + str(iou_v))

    def print_iou(self, x, y, n):
        iou = 0
        for i in range(n):
            mask = get_predictions(
                model, data[x][i], 640, 480, data['n_classes'], data['colormap'])
            iou += calculate_iou(data[y][i], mask)
        iou = iou/n
        return iou


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
    return gt, pred


def get_predictions(model, im, width, height, n_classes, colormap):
    input_data = []
    input_data.append(im)
    input_data = np.asarray(input_data)
    pred_mask = model.predict(input_data)
    pred_mask = tf.keras.backend.eval(pred_mask)[0]
    mask = np.zeros((height, width), dtype=np.int8)
    for i in range(n_classes):
        mask[pred_mask[:, :, i] >= 0.5] = i
    # mask = np.array(colormap)[mask].astype(np.uint8)
    # mask = mask[:, :, ::-1]
    return mask


def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def set_callbacks(path):
    log_dir = path + "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = TensorBoard(
        log_dir=log_dir, update_freq='batch', histogram_freq=1)

    checkpoint_path = path+'last_epoch/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    return [cp_callback, tensorboard_callback, IoUCallback()]


if __name__ == '__main__':
    data_file = 'data.h5'
    # bdd100k = BDD100k('/content/km10k/data/', 0.1, 640, 480)
    # data = bdd100k.data
    # class_weights = bdd100k.class_weights
    # n_classes = bdd100k.n_classes
    data = prepare_data(data_file=data_file, n_classes=7, valid_from_train=True,
                        n_valid=10, max_data=1000)
    data = shuffle_train_data(data)
    net = ERFNet([480, 640, 3], 7)
    model = net.model
    model.fit(data['x_train'],
              data['y_train'],
              epochs=50,
              validation_data=(data['x_val'], data['y_val']),
              validation_freq=10,
              class_weight=data['weights'],
              batch_size=8,
              callbacks=set_callbacks('/content/drive/My Drive/km10k/ERFNet/'))

    # model.evaluate((data['x_val'], data['y_val']))
