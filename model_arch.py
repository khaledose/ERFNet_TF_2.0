import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, concatenate, BatchNormalization, Dropout, add, Conv2DTranspose, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

tf.debugging.set_log_device_placement(True)


class MeanIoU(tf.keras.metrics.Metric):

    def __init__(self, name='miou', **kwargs):
        super(MeanIoU, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        smooth = 1
        iou = []
        for i in range(y_pred.shape[-1]):
            y = y_pred[:, :, :, i]
            intersection = tf.keras.backend.sum(
                tf.keras.backend.abs(y_true * y), axis=[1, 2, 3])

            union = tf.keras.backend.sum(y_true, [1, 2, 3]) + \
                tf.keras.backend.sum(y, [1, 2, 3])-intersection

            iou.append(tf.keras.backend.mean(
                (intersection + smooth) / (union + smooth), axis=0))
        return self.true_positives.assign_add(tf.keras.backend.mean(iou))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.)


class ERFNet:
    # ================================ INIT ===============================================
    def __init__(self, shape, num_classes):
        self.input_shape = shape
        self.trainMode = True
        self.num_classes = num_classes
        self.model = self._get_model()

    # ================================ GET MODEL ==========================================
    def _get_model(self, verbose=True):
        with tf.device('/device:GPU:1'):
            inputs = Input(shape=self.input_shape)
            x = self._downsampler_block(inputs, 16)
            x = self._downsampler_block(x, 64)
            x = self._non_bottleneck_1d(x, 64, 0.03, 1)
            x = self._non_bottleneck_1d(x, 64, 0.03, 1)
            x = self._non_bottleneck_1d(x, 64, 0.03, 1)
            x = self._non_bottleneck_1d(x, 64, 0.03, 1)
            x = self._non_bottleneck_1d(x, 64, 0.03, 1)
            x = self._downsampler_block(x, 128)
            x = self._non_bottleneck_1d(x, 128, 0.03, 2)
            x = self._non_bottleneck_1d(x, 128, 0.03, 4)
            x = self._non_bottleneck_1d(x, 128, 0.03, 8)
            x = self._non_bottleneck_1d(x, 128, 0.03, 16)
            x = self._non_bottleneck_1d(x, 128, 0.03, 2)
            x = self._non_bottleneck_1d(x, 128, 0.03, 4)
            x = self._non_bottleneck_1d(x, 128, 0.03, 8)
            x = self._non_bottleneck_1d(x, 128, 0.03, 16)
            x = self._upsampler_block(x, 64)
            x = self._non_bottleneck_1d(x, 64, 0.03, 1)
            x = self._non_bottleneck_1d(x, 64, 0.03, 1)
            x = self._upsampler_block(x, 16)
            x = self._non_bottleneck_1d(x, 16, 0.03, 1)
            x = self._non_bottleneck_1d(x, 16, 0.03, 1)
            x = Conv2DTranspose(filters=self.num_classes,
                                kernel_size=(2, 2), strides=2, padding='valid',  activation='softmax')(x)
            model = Model(inputs=inputs, outputs=x)

            optimizer = Adam(learning_rate=5e-4, beta_1=0.9,
                             beta_2=0.999, decay=2e-4, epsilon=1e-08)

            model.compile(optimizer=optimizer,
                          loss=SparseCategoricalCrossentropy(), metrics=None)

            if verbose:
                model.summary()

            return model

    # ================================ BOTTLENECK =========================================

    def _non_bottleneck_1d(self, layer, filter, dropout, dilRate):
        x = Conv2D(filters=filter, kernel_size=(
            3, 1), strides=1, padding='same')(layer)

        x = ReLU()(x)

        x = Conv2D(filters=filter, kernel_size=(
            1, 3), strides=1, padding='same')(x)

        x = BatchNormalization(epsilon=1e-03)(x)

        x = ReLU()(x)

        x = Conv2D(filters=filter, kernel_size=(3, 1), strides=1,
                   dilation_rate=(dilRate, 1), padding='same')(x)

        x = ReLU()(x)

        x = Conv2D(filters=filter, kernel_size=(1, 3), strides=1,
                   dilation_rate=(dilRate, 1), padding='same')(x)

        x = BatchNormalization(epsilon=1e-03)(x)

        x = Dropout(rate=dropout)(x)

        x = add([layer, x])

        x = ReLU()(x)
        return x

    # ================================ DOWNSAMPLER ========================================
    def _downsampler_block(self, layer, filter):
        shape = layer.shape
        outNo = int(shape[-1])
        x1 = Conv2D(filters=filter - outNo, kernel_size=(3, 3),
                    strides=2, padding='same')(layer)

        x2 = MaxPool2D(pool_size=(2, 2), strides=2)(layer)

        x = concatenate([x2, x1])

        x = ReLU()(x)
        return x

    def _upsampler_block(self, layer, filter):
        x = Conv2DTranspose(
            filter, (3, 3), strides=2, padding='same')(layer)
        x = BatchNormalization(epsilon=1e-03)(x)
        return ReLU()(x)
