import numpy as np
import tensorflow as tf
from model_arch import ERFNet
import cv2
import argparse


def load_model(checkpoint_path, width, height, n_classes):
    net = ERFNet([height, width, 3], n_classes)
    model = net.model
    model.load_weights(checkpoint_path)
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
    mask = np.array(colormap)[mask].astype(np.uint8)
    return mask[:, :, ::-1]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-mp",
                        help="set model path", type=str, default='./best_model/cp.ckpt')
    parser.add_argument("--image", "-i",
                        help="set input image", type=str, default='./input.jpg')
    parser.add_argument("--output_dir", "-o",
                        help="set output directory", type=str, default='./')
    args = parser.parse_args()

    width = 640
    height = 480
    n_classes = 7
    colormap = [[0, 0, 0], [255, 0, 0], [0, 255, 0], [
        0, 0, 255], [255, 255, 0], [255, 0, 255], [0, 255, 255]]
    checkpoint_path = args.model_path
    image_path = args.image
    output_path = args.output_dir

    im = cv2.imread(image_path)
    im = cv2.resize(im, (width, height))
    model = load_model(checkpoint_path, width, height, n_classes)
    mask = get_mask(model, im, width, height, n_classes, colormap)
    cv2.imwrite(output_path, mask)
