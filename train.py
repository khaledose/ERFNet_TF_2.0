import tensorflow as tf
from tensorflow.keras.callbacks import Callback
import argparse
from model_arch import ERFNet
from dataset import BDD100k, obj2h5
from visualizer import draw_training_curve, viz_segmentation_pairs
import numpy as np
import os
import math
import sys


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, n_samples, state):
        self.batch_size = batch_size
        self.n_samples = n_samples
        self.state = state
        self.dataset = dataset

    def __len__(self):
        return (np.ceil(self.n_samples / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        start, end = idx * self.batch_size, (idx+1) * self.batch_size
        data = self.dataset.prepare_batch('train', start, end)
        return np.array(data['x_'+self.state]), np.array(data['y_'+self.state])


class IoUCallback(Callback):
    def __init__(self, history, data, file, action, iou_data, model):
        self.history = history
        self.data = data
        self.file = file
        self.action = action
        self.model = model
        self.iou_data = iou_data

    def on_epoch_end(self, epoch, logs=None):
        if (epoch+1) % self.data['freq_iou'] != 0:
            return

        if self.action['train_iou']:
            iou_t = self.batch_iou('train', len(self.iou_data['x_train']))
            self.history['train_iou'] = np.append(
                self.history['train_iou'], iou_t)

        if self.action['val_iou']:
            iou_v = self.batch_iou('val', len(self.iou_data['x_train']))
            self.save_best_model(iou_v)
            self.history['val_iou'] = np.append(self.history['val_iou'], iou_v)

        if self.action['visualize']:
            self.draw_curves(self.history)

        self.history['epoch'] = np.append(self.history['epoch'], epoch)
        obj2h5(self.history, self.file['history'])

    def save_best_model(self, iou_v):
        iou = self.history['val_iou']
        if iou.shape[0] == 0 or iou_v > np.amax(iou):
            path = self.file['savedir']+'best_model/cp.ckpt'
            print("Saving best model at " + path)
            self.model.save_weights(path)

    def batch_iou(self, state, n_samples):
        iou = 0
        print("\n"+state+" IoU = ")
        for i in range(n_samples):
            mask = get_predictions(
                self.model, self.iou_data['x_'+state][i], self.data)
            mask = tf.keras.utils.to_categorical(
                y=mask, num_classes=self.data['n_classes'], dtype='uint8')
            iou += calculate_iou(self.iou_data['y_'+state]
                                 [i], mask, self.data['n_classes'])
            sys.stdout.write(str("\r {:.3%}").format(iou/(i+1)))
            sys.stdout.flush()
        print('\n')
        iou = iou/n_samples
        return iou

    def draw_curves(self, history):
        draw_training_curve(history['train_iou'], history['val_iou'],
                            self.file['savedir']+"iou.png", "IoU over time", "IoU", "lower right")


class VisualizationCallback(Callback):
    def __init__(self, file, viz, data, model):
        self.file = file
        self.viz = viz
        self.data = data
        self.model = model

    def on_epoch_end(self, epoch, logs=None):
        self.draw_samples(epoch, 'val')
        self.draw_samples(epoch, 'train')

    def draw_samples(self, epoch, state):
        preds_v = []
        viz_img_template = os.path.join(
            self.file['savedir'], "samples", "{}", "epoch_{: 07d}.jpg")
        for i in range(8):
            preds_v.append(get_predictions(
                self.model, self.viz['x_'+state][i], self.data))
        preds_v = np.asarray(preds_v)
        viz_segmentation_pairs(
            self.viz['x_'+state+'_viz'], self.viz['y_'+state+'_viz'], preds_v, self.viz['colormap'], (
                2, 4), viz_img_template.format(state, epoch))


def get_predictions(model, im, data):
    pred_mask = model.predict(np.array([im]))
    pred_mask = tf.keras.backend.eval(pred_mask)[0]
    pred_mask[pred_mask[:, :] < 0.5] = 0
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask = tf.keras.backend.eval(pred_mask)
    pred_mask = pred_mask[:, :, 0]
    return pred_mask


def calculate_iou(y_true, y_pred, n_classes):
    iou = np.zeros(n_classes)
    for i in range(1, n_classes):
        intersection = np.sum(np.logical_and(y_true[:, :, i], y_pred[:, :, i]))
        union = np.sum(np.logical_or(y_true[:, :, i], y_pred[:, :, i]))
        iou[i] += intersection/union if union != 0 else 0
    return np.sum(iou)/n_classes


def main(args):
    file = {}
    data = {}
    train_method = {}
    action = {}
    file['savedir'] = args.savedir
    file['datadir'] = './dataset/'
    file['data'] = file['savedir'] + 'dataset.h5'
    file['viz'] = file['savedir'] + 'viz.h5'
    file['history'] = file['savedir'] + "history.h5"
    file['weights'] = file['savedir'] + "weights.h5"
    file['iou'] = file['savedir'] + "iou.h5"
    file['checkpoint'] = file['savedir'] + 'last_epoch/cp.ckpt'
    data['width'] = args.width
    data['height'] = args.height
    data['train_limit'] = args.train_limit
    data['val_limit'] = args.val_limit
    data['n_epochs'] = args.epochs
    data['batch_size'] = args.batch_size
    data['val_from_train'] = args.valFromTrain
    data['freq_iou'] = args.freq_iou
    data['initial_epoch'] = 0
    data['n_classes'] = 7
    train_method['ram'] = args.ram
    train_method['disk'] = args.disk
    action['train_iou'] = args.train_iou
    action['val_iou'] = args.val_iou
    action['visualize'] = args.visualize

    dataset = BDD100k(file, data, train_method, action)
    history = dataset.history
    initial_weights = dataset.weights['weights']
    viz = dataset.viz
    if action['train_iou'] and action['val_iou']:
        iou_data = dataset.iou_data
    data['train_limit'] = dataset.train_samples
    data['val_limit'] = dataset.val_samples

    net = ERFNet([data['height'], data['width'], 3],
                 data['n_classes'])
    model = net.model

    if os.path.isfile(file['checkpoint']+'.index'):
        print("Loading weights from checkpoint")
        model.load_weights(file['checkpoint'])

    if len(history['epoch']) > 0:
        data['initial_epoch'] = int(history['epoch'][-1]) + 1

    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(file['checkpoint'],
                                                       save_weights_only=True,
                                                       verbose=1)
    callbacks = [checkpoint_cb]
    if action['train_iou'] and action['val_iou']:
        iou_cb = IoUCallback(history, data, file, action, iou_data, model)
        callbacks.append(iou_cb)
    if action['visualize']:
        viz_cb = VisualizationCallback(file, viz, data, model)
        callbacks.append(viz_cb)

    if train_method['ram']:
        inputs = dataset.inputs
        model.fit(inputs['x_train'],
                  inputs['y_train'],
                  epochs=data['n_epochs'],
                  verbose=1,
                  batch_size=data['batch_size'],
                  initial_epoch=data['initial_epoch'],
                  callbacks=callbacks)

    if train_method['disk']:
        train_batch_generator = BatchGenerator(
            dataset, data['batch_size'], data['train_limit'], 'train')
        model.fit(train_batch_generator,
                  epochs=data['n_epochs'],
                  verbose=1,
                  initial_epoch=data['initial_epoch'],
                  callbacks=callbacks)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--savedir",
                        help="sets training outputs' directory", type=str, default='./')
    parser.add_argument(
        "--width", help="sets network width", type=int, default=640)
    parser.add_argument(
        "--height", help="sets network height", type=int, default=480)
    parser.add_argument(
        "--train-limit", help="sets max number of training inputs", type=int, default=None)
    parser.add_argument(
        "--val-limit", help="sets max number of validation inputs", type=int, default=None)
    parser.add_argument(
        "--valFromTrain", help="splits dataset into train and val sets. \n MUST USE val-limit argument with this option", action="store_true")
    parser.add_argument(
        "--epochs", help="sets number of epochs", type=int, default=150)
    parser.add_argument(
        "--batch-size", help="sets training batch size", type=int, default=16)
    parser.add_argument(
        "--disk", help="loads batches of data from disk", action="store_true")
    parser.add_argument(
        "--ram", help="loads data from ram", action="store_true")
    parser.add_argument(
        "--train-iou", help="prints IoU of train data on each epoch", action="store_true")
    parser.add_argument(
        "--val-iou", help="prints IoU of validation data on each epoch", action="store_true")
    parser.add_argument(
        "--freq-iou", help="frequency of IoU calculation", type=int, default=1)
    parser.add_argument(
        "--visualize", help="plots IoU over time and shows comparisons between GT and predictions", action="store_true")

    main(parser.parse_args())
