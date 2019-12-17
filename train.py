import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from data_processing import prepare_data, obj2h5, h52obj, shuffle_train_data
from dataset import BDD100k
from model_arch import ERFNet
from tensorflow.keras.callbacks import TensorBoard
import datetime
import os


class MyCustomCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        print(epoch, logs)


def set_callbacks():
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tensorboard_callback = TensorBoard(
        log_dir=log_dir, update_freq='batch', histogram_freq=1)

    checkpoint_path = '/content/drive/My Drive/km10k/checkpoints/cp.ckpt'
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                     save_weights_only=True,
                                                     verbose=1)
    return [cp_callback, tensorboard_callback, MyCustomCallback()]


if __name__ == '__main__':
    data_file = 'data.h5'
    DATA_LIMIT = None
    # bdd100k = BDD100k('/content/km10k/data/', 0.1, 640, 480)
    # data = bdd100k.data
    # class_weights = bdd100k.class_weights
    # n_classes = bdd100k.n_classes
    data = h52obj(data_file)
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
              callbacks=set_callbacks())

    # model.evaluate((data['x_val'], data['y_val']))
