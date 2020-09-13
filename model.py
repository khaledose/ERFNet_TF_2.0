from utils import h52obj, obj2h5, colormap
from erfnet import ERFNet
import tensorflow as tf
import numpy as np
import os


class BatchGenerator(tf.keras.utils.Sequence):
    def __init__(self, dataset, batch_size, n_samples, shuffle=True):
        self.batch_size = batch_size
        self.dataset = dataset
        self.x_files = dataset.x_files
        self.y_files = dataset.y_files
        if shuffle:
            self.x_files, self.y_files = dataset.shuffle(
                x=self.x_files, y=self.y_files)
        self.n_samples = len(dataset) if n_samples is None else n_samples

    def __len__(self):
        return (np.ceil(self.n_samples / float(self.batch_size))).astype(np.int)

    def __getitem__(self, idx):
        start, end = idx * self.batch_size, (idx+1) * self.batch_size
        x = self.x_files[start:end]
        y = self.y_files[start:end]
        x, y = self.dataset.preprocess_images(x=x, y=y)
        return x, y


class Model():
    def __init__(self, shape, n_classes, isTraining=True, verbose=True):
        self.shape = shape
        self.n_classes = n_classes
        self.net = ERFNet(shape=shape, n_classes=n_classes,
                          isTraining=isTraining, verbose=verbose)
        self.model = self.net.model
        self.callbacks = []
        self.initial_epoch = 0
        self.state_dict = {}

    def update_state(self, current_state):
        for key, value in current_state.items():
            if key in self.state_dict:
                self.state_dict[key] = np.append(self.state_dict[key], value)
            else:
                self.state_dict[key] = np.array([value])

    def save_state(self, saveto):
        saveto = os.path.join(saveto, 'state.h5')
        obj2h5(self.state_dict, saveto)

    def load_state(self, path):
        try:
            print("Loading training state from HDF5")
            path = os.path.join(path, 'state.h5')
            self.state_dict = h52obj(path)
            self.initial_epoch = int(self.state_dict['epoch'][-1]) + 1
        except:
            print('Error! H5 not found!')

    def load_checkpoint(self, path):
        try:
            print("Loading weights from checkpoint")
            path = os.path.join(path, 'cp.ckpt')
            self.model.load_weights(path)
        except:
            print('Error! Checkpoint not found!')

    def save_checkpoint(self, path):
        print('Saving weights to checkpoint')
        path = os.path.join(path, 'cp.ckpt')
        self.model.save_weights(path)

    def add_callback(self, callback):
        self.callbacks.append(callback)

    def train(self, x, y, n_epochs, batch_size):
        self.model.fit(
            x=x,
            y=y,
            epochs=n_epochs,
            verbose=1,
            batch_size=batch_size,
            initial_epoch=self.initial_epoch,
            callbacks=self.callbacks,
        )

    def train_generator(self, generator, n_epochs, batch_size):
        self.model.fit(
            x=generator,
            epochs=n_epochs,
            verbose=1,
            initial_epoch=self.initial_epoch,
            callbacks=self.callbacks,
        )

    def evaluate(self, x, y, state):
        print('Evaluating model')
        iou = self.iou_on_batch(
            x=x,
            y=y,
        )
        print(state+' M-IoU = {:.3%}'.format(iou))
        return iou

    def predict(self, image, threshold=0.5):
        pred_mask = self.model.predict(np.array([image]))
        pred_mask = tf.keras.backend.eval(pred_mask)[0]
        pred_mask[pred_mask[:, :] < threshold] = 0
        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = np.asarray(pred_mask, dtype=np.uint8)
        return pred_mask
        # pred_mask = pred_mask[..., tf.newaxis]
        # pred_mask = tf.keras.backend.eval(pred_mask)
        # pred_mask = pred_mask[:, :, 0]
        # return pred_mask

    def iou_on_batch(self, x, y):
        iou = 0
        n_samples = len(x)
        for i in range(n_samples):
            y_pred = self.predict(
                image=x[i]
            )
            y_pred = tf.keras.utils.to_categorical(
                y=y_pred,
                num_classes=self.n_classes,
                dtype='uint8'
            )
            iou += self.calculate_iou(
                y_true=y[i],
                y_pred=y_pred,
            )
        return iou/n_samples

    def calculate_iou(self, y_true, y_pred):
        n_classes = self.n_classes
        iou = np.zeros(n_classes - 1)
        for i in range(1, n_classes):
            intersection = np.sum(np.logical_and(
                y_true[:, :, i], y_pred[:, :, i]))
            union = np.sum(np.logical_or(y_true[:, :, i], y_pred[:, :, i]))
            iou[i-1] += intersection/union if union != 0 else 0
        return np.average(iou)
