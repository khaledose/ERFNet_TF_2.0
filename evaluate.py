from model import Model
from dataset import BDD100k
import argparse
import os


def main(args):
    model_path = args.model_dir
    shape = (args.width, args.height)
    n_classes = args.num_classes
    n_samples = args.num_samples
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

    model = Model(
        shape=[args.height, args.width, 3],
        n_classes=n_classes,
        isTraining=False,
        verbose=False,
    )
    checkpoint_path = os.path.join(model_path)
    model.load_checkpoint(path=checkpoint_path)
    if n_samples is None:
        n_samples = len(val_ds)
    xtf, ytf = train_ds.shuffle(x=train_ds.x_files, y=train_ds.y_files)
    x, y = train_ds.preprocess_images(
        x=xtf[:n_samples],
        y=ytf[:n_samples],
        weighted_label=False
    )
    print(n_samples, 'Train data')
    model.evaluate(x=x, y=y)

    xvf, yvf = val_ds.shuffle(x=val_ds.x_files, y=val_ds.y_files)
    x, y = val_ds.preprocess_images(
        x=xvf[:n_samples],
        y=yvf[:n_samples],
        weighted_label=False
    )
    print(n_samples, 'Validation data')
    model.evaluate(x=x, y=y)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-dir", help="sets training outputs' directory", type=str, default='./')
    parser.add_argument(
        "--datadir", help="sets dataset directory", type=str, default='./dataset/')
    parser.add_argument(
        "--width", help="sets network width", type=int, default=640)
    parser.add_argument(
        "--height", help="sets network height", type=int, default=480)
    parser.add_argument(
        "--num-classes", help="sets number of classes", type=int, default=7)
    parser.add_argument(
        "--num-samples", help="sets number of evaluation samples", type=int, default=None)

    main(parser.parse_args())
