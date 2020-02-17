import numpy as np
import tensorflow as tf
from model_arch import ERFNet
import cv2
import argparse
import PIL


def load_model(checkpoint_path, width, height, n_classes):
    net = ERFNet([height, width, 3], n_classes)
    model = net.model
    model.load_weights(checkpoint_path)
    return model


def get_predictions(model, im, colormap):
    pred_mask = model.predict(np.array([im]))
    pred_mask = tf.keras.backend.eval(pred_mask)[0]
    pred_mask[pred_mask[:, :] < 0.5] = 0
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    pred_mask = tf.keras.backend.eval(pred_mask)
    pred_mask = pred_mask[:, :, 0]
    mask = np.array(colormap)[pred_mask].astype(np.uint8)
    return mask


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

    img = PIL.Image.open(image_path).resize(
        [width, height], resample=PIL.Image.CUBIC)
    img = np.asarray(img, dtype=np.float32)
    img = np.divide(img, 255.0)
    model = load_model(checkpoint_path, width, height, n_classes)
    mask = get_predictions(model, img, colormap)
    cv2.imwrite(output_path, mask)
