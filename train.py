import argparse
import datetime
from model_arch import ERFNet
from dataset import BDD100k, obj2h5, h52obj, get_data
from visualizer import draw_training_curve, viz_segmentation_pairs
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import os


class BatchGenerator(tf.keras.utils.Sequence):

    def __init__(self, data_h5, batch_size, n_samples, state):
        self.data_h5 = data_h5
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.state = state

    def __len__(self):
        return (np.ceil(self.n_samples / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        mini, maxi = idx * self.batch_size, (idx+1) * self.batch_size
        data = h52obj(self.data_h5, mini, maxi)
        batch_x = data['x_'+self.state]
        batch_y = data['y_'+self.state]
        return np.array(batch_x), np.array(batch_y)


class HistoryCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        iou_t = self.print_iou('x_train', 'y_train', val_split)

        if train_method == 1:
            iou_v = self.print_iou('x_val', 'y_val', val_split)
            self.save_best_model(iou_v)
            print("Epoch " + str(epoch+1) + "Validation IoU= " + str(iou_v))
            history['val_iou'] = np.append(history['val_iou'], iou_v)
            self.draw_samples(epoch, 'val')

        print("Epoch "+str(epoch+1)+": Train IoU = " + str(iou_t))
        history['train_iou'] = np.append(history['train_iou'], iou_t)
        history['epoch'] = np.append(history['epoch'], epoch)
        obj2h5(history, history_file)
        self.draw_samples(epoch, 'train')

    def draw_curves(self, history):
        draw_training_curve(history['train_loss'], history['val_loss'],
                            model_path+"loss.png", "Loss over time", "Loss", "lower right")
        draw_training_curve(history['train_iou'], history['val_iou'],
                            model_path+"iou.png", "IoU over time", "IoU", "lower right")

    def save_best_model(self, iou_v):
        iou = history['val_iou']
        if iou.shape[0] == 0 or iou_v > np.amax(iou):
            model.save_weights(model_path+'best_model/cp.ckpt')

    def draw_samples(self, epoch, x):
        preds_v = []
        viz_img_template = os.path.join(
            model_path, "samples", "{}", "epoch_{: 07d}.jpg")
        for i in range(8):
            preds_v.append(get_predictions(
                model, data['x_'+x][i], width, height, data['n_classes'][0], data['colormap']))
        preds_v = np.asarray(preds_v)
        viz_segmentation_pairs(
            data['x_'+x][:8], data['y_'+x][:8], preds_v, data['colormap'], (
                2, 4), viz_img_template.format(x, epoch))

    def print_iou(self, x, y, n):
        iou = 0
        for i in range(n):
            mask = get_predictions(
                model, data[x][i], width, height, data['n_classes'][0], data['colormap'])
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


def prepare_history(file):
    data = {}
    data['train_iou'] = []
    data['val_iou'] = []
    data['epoch'] = []

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-mp",
                        help="set training directory", type=str, default='./')
    parser.add_argument(
        "--width", "-w", help="set network width", type=int, default=640)
    parser.add_argument(
        "--height", "-ht", help="set network height", type=int, default=480)
    parser.add_argument(
        "--limit", "-l", help="set dataset inputs limit", type=int, default=20000)
    parser.add_argument(
        "--vlimit", "-v", help="set val split", type=int, default=1000)
    parser.add_argument(
        "--epoch", "-e", help="set training number of epochs", type=int, default=150)
    parser.add_argument(
        "--batch", "-b", help="set training batch size", type=int, default=16)
    parser.add_argument(
        "--train_method", "-t", help="train on full data or train by batch", type=int, default=1)
    args = parser.parse_args()
    model_path = args.model_path
    colab_path = '/content/ERFNet_TF_2.0/'
    data_dir = colab_path + "dataset/"
    data_h5 = data_dir + 'data.h5'
    history_file = model_path + "history.h5"
    width, height = args.width, args.height
    data_limit = args.limit
    val_split = args.vlimit
    n_epochs = args.epoch
    batch_size = args.batch
    train_method = args.train_method
    if not os.path.isfile(history_file):
        prepare_history(history_file)
    history = h52obj(history_file)
    dataset = BDD100k(data_dir, width, height, data_limit,
                      val_split, 7, train_method)
    data = {}
    if train_method == 0:
        data = dataset.data
        data = dataset.shuffle_train_data(data)
        net = ERFNet([height, width, 3], data['n_classes'][0])
        model = net.model
        model.fit(data['x_train'],
                  data['y_train'],
                  epochs=n_epochs,
                  class_weight=data['weights'],
                  batch_size=batch_size,
                  callbacks=set_callbacks(model_path))
    else:
        train_batch_generator = BatchGenerator(
            data_h5, batch_size, 900, 'train')
        # val_batch_generator = BatchGenerator(
        #     data_h5, batch_size, val_split, 'val')
        data = get_data(data_h5)
        net = ERFNet([height, width, 3], data['n_classes'][0])
        model = net.model
        model.fit_generator(generator=train_batch_generator,
                            # validation_data=val_batch_generator,
                            # validation_freq=1,
                            epochs=n_epochs,
                            verbose=1,
                            class_weight=data['weights'],
                            callbacks=set_callbacks(model_path))
