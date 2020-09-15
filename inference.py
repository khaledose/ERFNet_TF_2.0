from visualizer import get_mask
from utils import h52obj
from model import Model
import numpy as np
import argparse
import glob
import PIL
import cv2
import os


def preprocess_image(image):
    img = PIL.Image.open(image).resize(
        [width, height], resample=PIL.Image.CUBIC)
    img = np.asarray(img, dtype=np.float32)
    img = np.divide(img, 255.0)
    return img


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir",
                        help="set model path", type=str, default='./model/cp.ckpt')
    parser.add_argument("--image",
                        help="set input image", type=str, default=None)
    parser.add_argument("--image-dir",
                        help="set directory of set of input images", type=str, default=None)
    parser.add_argument("--output-dir",
                        help="set output directory", type=str, default='./')
    parser.add_argument("--width",
                        help="set image width", type=int, default=640)
    parser.add_argument("--height",
                        help="set image height", type=int, default=480)
    parser.add_argument("--num-classes",
                        help="set number of classes", type=int, default=7)
    parser.add_argument("--threshold",
                        help="set number of classes", type=float, default=0.5)
    args = parser.parse_args()

    width = args.width
    height = args.height
    n_classes = args.num_classes
    checkpoint_path = args.model_dir + '/last_epoch/cp.ckpt'
    output_path = args.output_dir
    colormap = h52obj(os.path.join(args.model_dir, 'colormap.h5'))['colormap']
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    model = Model(
        shape=[height, width, 3],
        n_classes=n_classes,
        isTraining=False,
        verbose=False
    )
    model.load_checkpoint(path=checkpoint_path)
    if args.image is not None:
        images = [args.image]
    if args.image_dir is not None:
        images = glob.glob(os.path.join(args.image_dir, "*.jpg"))

    for i in range(len(images)):
        image = preprocess_image(image=images[i])
        prediction = model.predict(image=image, threshold=args.threshold)
        mask = get_mask(mask=prediction, colormap=colormap)
        op = os.path.join(output_path, 'output_'+str(i+1)+'.png')
        cv2.imwrite(op, mask)
