import numpy as np
import tensorflow as tf
from data_processing import prepare_data, shuffle_train_data, prepare_history, h52obj, obj2h5
from visualizer import draw_training_curve
from model_arch import ERFNet
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
import cv2


class HistoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        iou_t = self.print_iou('x_train', 'y_train', 10)
        iou_v = self.print_iou('x_val', 'y_val', 10)
        print("Epoch "+str(epoch+1)+": Train IoU = " +
              str(iou_t), "Validation IoU of  = " + str(iou_v))
        np.append(history['val_loss'], logs['val_loss'])
        np.append(history['train_loss'], logs['train_loss'])
        np.append(history['train_iou'], logs['train_iou'])
        np.append(history['val_iou'], logs['val_iou'])
        obj2h5(history, history_file)
        draw_training_curve(history['train_loss'], history['val_loss'],
                            "/content/drive/My Drive/km10k/ERFNet/loss.png", "Loss over time", "Loss", "lower right")
        draw_training_curve(history['train_iou'], history['val_iou'],
                            "/content/drive/My Drive/km10k/ERFNet/loss.png", "IoU over time", "IoU", "lower right")

    def print_iou(self, x, y, n):
        iou = 0
        for i in range(n):
            mask = get_predictions(
                model, data[x][i], 640, 480, data['n_classes'], data['colormap'])
            iou += calculate_iou(data[y][i], mask)
        iou = iou/n
        return iou


def get_predictions(model, im, width, height, n_classes, colormap):
    input_data = []
    input_data.append(im)
    input_data = np.asarray(input_data)
    pred_mask = model.predict(input_data)
    pred_mask = tf.keras.backend.eval(pred_mask)[0]
    mask = np.zeros((height, width), dtype=np.int8)
    for i in range(n_classes):
        mask[pred_mask[:, :, i] >= 0.5] = i
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
    return [cp_callback, tensorboard_callback, HistoryCallback()]


if __name__ == '__main__':
    data_file = 'data.h5'
    history_file = "/content/drive/My Drive/km10k/ERFNet/history.h5"
    # bdd100k = BDD100k('/content/km10k/data/', 0.1, 640, 480)
    # data = bdd100k.data
    # class_weights = bdd100k.class_weights
    # n_classes = bdd100k.n_classes
    data = prepare_data(data_file=data_file, n_classes=7, valid_from_train=True,
                        n_valid=10, max_data=100)
    history = h52obj(history_file)
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
