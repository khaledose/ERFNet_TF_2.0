import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, concatenate, BatchNormalization, Dropout, add, Conv2DTranspose, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


class ERFNet:
    # ================================ INIT ===============================================
    def __init__(self, shape, num_classes, isTraining=True):
        self.input_shape = shape
        self.trainMode = isTraining
        self.num_classes = num_classes
        self.model = self._get_model()

    # ================================ GET MODEL ==========================================
    def _get_model(self, verbose=True):
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
                      loss=CategoricalCrossentropy(), metrics=None)

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

        if self.trainMode:
            x = Dropout(rate=dropout)(x)
            x = add([x, layer])
        else:
            x = add([x, layer])

        return ReLU()(x)

    # ================================ DOWNSAMPLER ========================================
    def _downsampler_block(self, layer, filter):
        shape = layer.shape
        outNo = int(shape[-1])
        x1 = Conv2D(filters=filter - outNo, kernel_size=(3, 3),
                    strides=2, padding='same')(layer)

        x2 = MaxPool2D(pool_size=(2, 2), strides=2)(layer)

        x = concatenate([x2, x1])

        x = BatchNormalization(epsilon=1e-03)(x)

        return ReLU()(x)

    def _upsampler_block(self, layer, filter):
        x = Conv2DTranspose(
            filter, (3, 3), strides=2, padding='same')(layer)
        x = BatchNormalization(epsilon=1e-03)(x)
        return ReLU()(x)
