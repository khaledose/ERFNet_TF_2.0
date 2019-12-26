import argparse
import datetime
from model_arch import ERFNet
from dataset import BDD100k, obj2h5, h52obj
from visualizer import draw_training_curve, viz_segmentation_pairs
import numpy as np
from tensorflow.keras.callbacks import TensorBoard
import tensorflow as tf
import os


def on_epoch_end(epoch, logs=None):
    iou_v = print_iou('x_val', 'y_val', val_split)
    save_best_model(iou_v)
    print("Epoch " + str(epoch+1) + "Validation IoU= " + str(iou_v))
    history['val_iou'] = np.append(history['val_iou'], iou_v)
    obj2h5(history, history_file)
    draw_samples(epoch)


def save_best_model(iou_v):
    iou = history['val_iou']
    if iou.shape[0] == 0 or iou_v > np.amax(iou):
        model.save_weights(model_path+'best_model/cp.ckpt')


def draw_samples(epoch):
    preds_v = []
    viz_img_template = os.path.join(
        model_path, "samples", "{}", "epoch_{: 07d}.jpg")
    for i in range(8):
        preds_v.append(get_predictions(
            model, data['x_val'][i], width, height, data['n_classes'], data['colormap']))
    preds_v = np.asarray(preds_v)
    viz_segmentation_pairs(
        data['x_val'][:8], data['y_val'][:8], preds_v, data['colormap'], (
            2, 4), viz_img_template.format("val", epoch))


def print_iou(x, y, n):
    iou = 0
    for i in range(n):
        mask = get_predictions(
            model, data[x][i], width, height, data['n_classes'], data['colormap'])
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", "-mp",
                        help="set training directory", type=str, default='./')
    parser.add_argument(
        "--width", "-w", help="set network width", type=int, default=640)
    parser.add_argument(
        "--height", "-ht", help="set network height", type=int, default=480)
    parser.add_argument(
        "--limit", "-l", help="set dataset inputs limit", type=int, default=1000)
    parser.add_argument(
        "--vlimit", "-v", help="set val split", type=int, default=1000)
    parser.add_argument(
        "--epoch", "-e", help="set training number of epochs", type=int, default=150)
    parser.add_argument(
        "--batch", "-b", help="set training batch size", type=int, default=8)
    args = parser.parse_args()
    model_path = args.model_path
    colab_path = '/content/ERFNet_TF_2.0/'
    data_dir = colab_path + "dataset/"
    history_file = model_path + "history.h5"
    width, height = args.width, args.height
    data_limit = args.limit
    val_split = args.vlimit
    n_epochs = args.epoch
    batch_size = args.batch

    dataset = BDD100k(data_dir, width, height, data_limit, val_split, 7)
    data = dataset.data
    net = ERFNet([height, width, 3], data['n_classes'])
    model = net.model

    current_epoch = 0
    while True:
        history = h52obj(history_file)
        if history['epoch'][-1] == current_epoch:
            model.load_weights(model_path+'checkpoints/cp.ckpt')
            on_epoch_end(current_epoch)
            current_epoch += 1
