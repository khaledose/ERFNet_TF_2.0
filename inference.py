import numpy as np
import tensorflow as tf
from model_arch import ERFNet
import cv2


width = 640
height = 480
n_classes = 7
colormap = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [
    0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
checkpoint_path = '/content/drive/My Drive/km10k/checkpoints/cp.ckpt'
image_path = '/content/00adbb3f-7757d4ea.jpg'


def load_model(checkpoint_path, width, height, n_classes):
    net = ERFNet([height, width, 3], n_classes)
    model = net.model
    model.load_weights('/content/drive/My Drive/km10k/checkpoints/cp.ckpt')
    return model


def get_mask(model, im, width, height, n_classes, colormap):
    input_data = []
    input_data.append(im)
    input_data = np.asarray(input_data)
    pred_mask = model.predict(input_data)
    pred_mask = tf.keras.backend.eval(pred_mask)[0]
    mask = np.zeros((height, width), dtype=np.int8)
    for i in range(n_classes):
        mask[pred_mask[:, :, i] >= 0.5] = i
    return np.array(colormap)[mask].astype(np.uint8)


im = cv2.imread(image_path)
im = cv2.resize(im, (width, height))
model = load_model(checkpoint_path, width, height, n_classes)
mask = get_mask(model, im, width, height, n_classes, colormap)

cv2.imwrite('/content/output.png', mask)
