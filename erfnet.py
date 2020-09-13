import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, concatenate, BatchNormalization, Dropout, add, Conv2DTranspose, ReLU
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import CategoricalCrossentropy


class ERFNet:
    # ================================ INIT ===============================================
    def __init__(self, shape, n_classes, isTraining=True, verbose=True):
        self.input_shape = shape
        self.trainMode = isTraining
        self.n_classes = n_classes
        self.model = self._get_model(verbose)
    # ================================ GET MODEL ==========================================

    def _get_model(self, verbose=True):
        inputs = Input(
            shape=self.input_shape,
        )

        x = self._downsampler_block(
            layer=inputs,
            filter=16,
        )

        x = self._downsampler_block(
            layer=x,
            filter=64,
        )

        for i in range(5):
            x = self._non_bottleneck_1d(
                layer=x,
                filter=64,
                dropout=0.03,
                dilRate=1,
            )

        x = self._downsampler_block(
            layer=x,
            filter=128,
        )

        for i in range(1, 5):
            x = self._non_bottleneck_1d(
                layer=x,
                filter=128,
                dropout=0.03,
                dilRate=2**(i),
            )

        for i in range(1, 5):
            x = self._non_bottleneck_1d(
                layer=x,
                filter=128,
                dropout=0.03,
                dilRate=2**(i),
            )

        x = self._upsampler_block(
            layer=x,
            filter=64,
        )

        for i in range(2):
            x = self._non_bottleneck_1d(
                layer=x,
                filter=64,
                dropout=0.03,
                dilRate=1,
            )

        x = self._upsampler_block(
            layer=x,
            filter=16,
        )

        for i in range(2):
            x = self._non_bottleneck_1d(
                layer=x,
                filter=16,
                dropout=0.03,
                dilRate=1,
            )

        x = Conv2DTranspose(
            filters=self.n_classes,
            kernel_size=(2, 2),
            strides=2,
            padding='valid',
            activation='softmax',
        )(x)

        model = Model(
            inputs=inputs,
            outputs=x,
        )

        optimizer = Adam(
            learning_rate=5e-4,
            beta_1=0.9,
            beta_2=0.999,
            decay=2e-4,
            epsilon=1e-08,
        )

        model.compile(
            optimizer=optimizer,
            loss=CategoricalCrossentropy(),
            metrics=None,
        )

        if verbose:
            model.summary()

        return model
    # ================================ BOTTLENECK =========================================

    def _non_bottleneck_1d(self, layer, filter, dropout, dilRate):
        x = Conv2D(
            filters=filter,
            kernel_size=(3, 1),
            strides=1,
            padding='same',
        )(layer)

        x = ReLU()(x)

        x = Conv2D(
            filters=filter,
            kernel_size=(1, 3),
            strides=1,
            padding='same',
        )(x)

        x = BatchNormalization(
            epsilon=1e-03,
        )(x)

        x = ReLU()(x)

        x = Conv2D(
            filters=filter,
            kernel_size=(3, 1),
            strides=1,
            dilation_rate=(dilRate, 1),
            padding='same',
        )(x)

        x = ReLU()(x)

        x = Conv2D(
            filters=filter,
            kernel_size=(1, 3),
            strides=1,
            dilation_rate=(dilRate, 1),
            padding='same',
        )(x)

        x = BatchNormalization(
            epsilon=1e-03,
        )(x)

        if self.trainMode:
            x = Dropout(
                rate=dropout,
            )(x)
            x = add([x, layer])
        else:
            x = add([x, layer])

        return ReLU()(x)
    # ================================ DOWNSAMPLER ========================================

    def _downsampler_block(self, layer, filter):
        shape = layer.shape
        outNo = int(shape[-1])
        x1 = Conv2D(
            filters=filter - outNo,
            kernel_size=(3, 3),
            strides=2,
            padding='same',
        )(layer)

        x2 = MaxPool2D(
            pool_size=(2, 2),
            strides=2,
        )(layer)

        x = concatenate([x2, x1])

        x = BatchNormalization(
            epsilon=1e-03,
        )(x)

        return ReLU()(x)
    # ================================ UPSAMPLER ========================================

    def _upsampler_block(self, layer, filter):
        x = Conv2DTranspose(
            filters=filter,
            kernel_size=(3, 3),
            strides=2,
            padding='same',
        )(layer)

        x = BatchNormalization(
            epsilon=1e-03,
        )(x)
        return ReLU()(x)
