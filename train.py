from tensorflow.keras.callbacks import Callback, ModelCheckpoint
from visualizer import draw_training_curve, draw_samples
from model import Model, BatchGenerator
from dataset import BDD100k
from utils import get_colors, obj2h5, h52obj
import numpy as np
import argparse
import os


class IoUCallback(Callback):
    def __init__(self, dataset, model, state, n_samples=None):
        self.net = model
        self.state = state
        self.dataset = dataset
        self.x, self.y = dataset.preprocess_images(
            x=dataset.x_files[:n_samples],
            y=dataset.y_files[:n_samples],
            weighted_label=False
        )

    def on_epoch_end(self, epoch, logs=None):
        iou = self.net.evaluate(x=self.x, y=self.y, state=self.state)
        current_state = {self.state+'_iou': iou}
        self.net.update_state(current_state=current_state)


class VisualizationCallback(Callback):
    def __init__(self, model, dataset, state, saveto):
        self.x, self.y = dataset.preprocess_images(
            x=dataset.x_files[:8],
            y=dataset.y_files[:8],
            weighted_label=False,
            hot_label=False,
            normalize=False,
        )
        self.net = model
        self.state = state
        self.saveto = saveto
        self.colormap = dataset.colormap

    def on_epoch_end(self, epoch, logs=None):
        y_pred = np.array([self.net.predict(np.divide(im, 255.0)) for im in self.x])
        draw_samples([self.x, self.y, y_pred], self.colormap,
                     epoch, self.state, self.saveto)


class StateCallback(Callback):
    def __init__(self, model, saveto):
        self.net = model
        self.saveto = saveto

    def on_epoch_end(self, epoch, logs=None):
        current_state = {'epoch': epoch}
        self.net.update_state(current_state=current_state)
        self.net.save_state(saveto=self.saveto)


def main(args):
    model_path = args.savedir
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    shape = (args.width, args.height)
    n_classes = args.num_classes

    train_ds = BDD100k(
        datadir=args.datadir,
        shape=shape,
        n_classes=n_classes,
        state='train',
    )
    train_ds.load_images()

    val_ds = BDD100k(
        datadir=args.datadir,
        shape=shape,
        n_classes=n_classes,
        state='val',
    )
    val_ds.load_images()

    if not os.path.isfile(os.path.join(model_path, 'colormap.h5')):
        colormap = get_colors(val_ds.y_files, val_ds.n_classes)
        obj2h5({'colormap':colormap}, os.path.join(model_path, 'colormap.h5'))
    else:
        colormap = h52obj(os.path.join(model_path, 'colormap.h5'))['colormap']

    train_ds.colormap = colormap
    val_ds.colormap = colormap

    if not os.path.isfile(os.path.join(model_path, 'weights.h5')):
        train_ds.calculate_class_weights(y=train_ds.y_files[:args.train_limit])
        train_ds.save_weights(saveto=model_path)
    else:
        train_ds.load_weights(path=model_path)

    model = Model(
        shape=[args.height, args.width, 3],
        n_classes=n_classes,
        isTraining=True,
    )
    checkpoint_path = os.path.join(model_path, 'last_epoch')
    if args.resume:
        model.load_checkpoint(path=checkpoint_path)
        model.load_state(model_path)

    checkpoint_cb = ModelCheckpoint(
        os.path.join(checkpoint_path, 'cp.ckpt'),
        save_weights_only=True,
        verbose=1,
    )
    model.add_callback(checkpoint_cb)

    if args.iou_train:
        train_iou_cb = IoUCallback(
            dataset=train_ds,
            model=model,
            state='train',
            n_samples=args.val_limit if args.val_limit is not None else len(
                val_ds)
        )
        model.add_callback(train_iou_cb)
    if args.iou_val:
        val_iou_cb = IoUCallback(
            dataset=val_ds,
            model=model,
            state='val',
            n_samples=args.val_limit if args.val_limit is not None else len(
                val_ds)
        )
        model.add_callback(val_iou_cb)
    if args.visualize:
        viz_train_callback = VisualizationCallback(
            model=model, dataset=train_ds, state='train', saveto=model_path)
        viz_val_callback = VisualizationCallback(
            model=model, dataset=val_ds, state='val', saveto=model_path)
        model.add_callback(viz_train_callback)
        model.add_callback(viz_val_callback)

    state_cb = StateCallback(model=model, saveto=model_path)
    model.add_callback(state_cb)

    if args.ram:
        x_train, y_train = train_ds.shuffle(
            x=train_ds.x_files, y=train_ds.y_files)
        x_train, y_train = train_ds.preprocess_images(
            x=x_train[:args.train_limit],
            y=y_train[:args.train_limit],
            normalize=True,
            hot_label=True,
            weighted_label=True,
        )
        model.train(
            x=x_train,
            y=y_train,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
        )
    if args.disk:
        generator = BatchGenerator(
            dataset=train_ds, batch_size=args.batch_size, n_samples=args.train_limit)
        model.train_generator(
            generator=generator,
            n_epochs=args.epochs,
            batch_size=args.batch_size,
        )
    print('Training Finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--savedir", help="sets training outputs' directory", type=str, default='./')
    parser.add_argument(
        "--datadir", help="sets dataset directory", type=str, default='./dataset/')
    parser.add_argument(
        "--width", help="sets network width", type=int, default=640)
    parser.add_argument(
        "--height", help="sets network height", type=int, default=480)
    parser.add_argument(
        "--num-classes", help="sets number of classes", type=int, default=7)
    parser.add_argument(
        "--train-limit", help="sets max number of training inputs", type=int, default=None)
    parser.add_argument(
        "--val-limit", help="sets max number of validation inputs", type=int, default=None)
    parser.add_argument(
        "--epochs", help="sets number of epochs", type=int, default=150)
    parser.add_argument(
        "--batch-size", help="sets training batch size", type=int, default=16)
    parser.add_argument(
        "--disk", help="loads batches of data from disk", action="store_true")
    parser.add_argument(
        "--ram", help="loads data from ram", action="store_true")
    parser.add_argument(
        "--iou-train", help="prints IoU of train data on each epoch", action="store_true")
    parser.add_argument(
        "--iou-val", help="prints IoU of validation data on each epoch", action="store_true")
    parser.add_argument(
        "--visualize", help="plots IoU over time and shows comparisons between GT and predictions", action="store_true")
    parser.add_argument(
        "--resume", help="resumes training from checkpoint", action="store_true")

    main(parser.parse_args())
