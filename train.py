import numpy as np
import tensorflow as tf
from visualizer import draw_training_curve, viz_segmentation_pairs
from dataset import BDD100k, obj2h5, h52obj
from model_arch import ERFNet
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os
import cv2


class HistoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        iou_t = self.print_iou('x_train', 'y_train', val_split)
        iou_v = self.print_iou('x_val', 'y_val', val_split)
        self.save_best_model(iou_v)
        print("Epoch "+str(epoch+1)+": Train IoU = " +
              str(iou_t), "Validation IoU of  = " + str(iou_v))
        if 'val_loss' in logs:
            history['val_loss'] = np.append(
                history['val_loss'], logs['val_loss'])
            history['train_loss'] = np.append(
                history['train_loss'], logs['loss'])
        history['train_iou'] = np.append(history['train_iou'], iou_t)
        history['val_iou'] = np.append(history['val_iou'], iou_v)
        obj2h5(history, history_file)
        self.draw_curves(history)
        self.draw_samples(epoch)

    def draw_curves(self, history):
        draw_training_curve(history['train_loss'], history['val_loss'],
                            model_path+"loss.png", "Loss over time", "Loss", "lower right")
        draw_training_curve(history['train_iou'], history['val_iou'],
                            model_path+"iou.png", "IoU over time", "IoU", "lower right")

    def save_best_model(self, iou_v):
        iou = history['val_iou']
        if iou.shape[0] == 0 or iou_v > np.amax(iou):
            model.save_weights(model_path+'best_model/cp.ckpt')

    def draw_samples(self, epoch):
        preds_t = []
        preds_v = []
        viz_img_template = os.path.join(
            model_path, "samples", "{}", "epoch_{: 07d}.jpg")
        for i in range(8):
            preds_t.append(get_predictions(
                model, data['x_train_viz'][i], width, height, data['n_classes'], data['colormap']))
            preds_v.append(get_predictions(
                model, data['x_val'][i], width, height, data['n_classes'], data['colormap']))
        preds_t = np.asarray(preds_t)
        preds_v = np.asarray(preds_v)
        viz_segmentation_pairs(
            data['x_train_viz'][:8], data['y_train_viz'][:8], preds_t, data['colormap'], (
                2, 4), viz_img_template.format("train", epoch))
        viz_segmentation_pairs(
            data['x_val'][:8], data['y_val'][:8], preds_v, data['colormap'], (
                2, 4), viz_img_template.format("val", epoch))

    def print_iou(self, x, y, n):
        iou = 0
        for i in range(n):
            mask = get_predictions(
                model, data[x][i], width, height, data['n_classes'], data['colormap'])
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
        mask[pred_mask[:, :, i] >= 0.005] = i
    return mask


def calculate_iou(y_true, y_pred):
    intersection = np.logical_and(y_true, y_pred)
    union = np.logical_or(y_true, y_pred)
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score


def prepare_history(file):
    data = {}
    data['train_loss'] = []
    data['val_loss'] = []
    data['train_iou'] = []
    data['val_iou'] = []

    obj2h5(data, file)


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
    model_path = "/content/drive/My Drive/km10k/ERFNet/"
    data_dir = "/content/ERFNet_TF_2.0/km10k/"
    history_file = model_path + "history.h5"
    width, height = 640, 480
    data_limit = 1000
    val_split = 100
    n_epochs = 50
    batch_size = 8

    if not os.path.isfile(history_file):
        prepare_history(history_file)
    history = h52obj(history_file)

    dataset = BDD100k(data_dir, width, height, data_limit, val_split, 7)
    data = dataset.data
    data = dataset.shuffle_train_data(data)
    net = ERFNet([height, width, 3], data['n_classes'])
    model = net.model
    model.fit(data['x_train'],
              data['y_train'],
              epochs=n_epochs,
              validation_data=(data['x_val'], data['y_val']),
              validation_freq=5,
              class_weight=data['weights'],
              batch_size=batch_size,
              callbacks=set_callbacks(model_path))
