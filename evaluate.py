import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time
import shutil
import argparse
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from tqdm import tqdm
from DataPreprocess import ImageReader,IMG_MEAN
from fcn_mil import MIL_FCN

# model parameters
input_size = '256,256'
batch_size = 32
num_classes = 21

# train datasets and validation datasets
data_directory = './Datasets/VOCdevkit'
train_data_list_path = './dataset/train.txt'
val_data_list_path = './dataset/val.txt'

# optimizer parameters
learning_rate = 1e-4
momentum = 0.9
weight_decay = 5e-4

num_images = 1400

save_figure_dir = './val_images'
weights_path = './MIL-FCN/MIL_FCN_1000.h5'


def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="MIL-FCN8VGG16 Network")
    parser.add_argument("--batch_size", type=int, default=batch_size,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--num_classes", type=int, default=num_classes,
                        help="Number of images classes.")
    parser.add_argument("--data_dir", type=str, default=data_directory,
                        help="Path to the directory containing the PASCAL VOC dataset.")
    parser.add_argument("--train_data_list", type=str, default=train_data_list_path,
                        help="Path to the file listing the images in the training set.")
    parser.add_argument("--val_data_list", type=str, default=val_data_list_path,
                        help="Path to the file listing the images in the validation set.")
    parser.add_argument("--input_size", type=str, default=input_size,
                        help="Comma-separated string with height and width of images.")
    parser.add_argument("--learning_rate", type=float, default=learning_rate,
                        help="Learning rate for training.")
    parser.add_argument("--momentum", type=float, default=momentum,
                        help="Momentum parameter")
    parser.add_argument("--weight_decay", type=float, default=weight_decay,
                        help="Weight decay parameter")
    parser.add_argument("--num_images", type=int, default=num_images,
                        help="Number of images in the validation set.")
    parser.add_argument("--save_figure_dir", type=str, default=save_figure_dir,
                        help="Where to save figures with predictions.")
    parser.add_argument("--weights_path", type=str, default=weights_path,
                        help="Path to the file with model weights. "
                             "If not set, all the variables are initialised randomly.")

    return parser.parse_args()

def evaluate():

    args = get_arguments()
    h, w = map(int, args.input_size.split(','))

    if args.save_figure_dir is not None:
        if os.path.exists(args.save_figure_dir):
            shutil.rmtree(args.save_figure_dir)
            os.mkdir(args.save_figure_dir)
        else :
            os.mkdir(args.save_figure_dir)

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=args.momentum,
                                         beta_2=0.996, decay=args.weight_decay)

    model = MIL_FCN(args.num_classes, args.batch_size, optimizer)
    model.build((None, h, w, 3))
    if args.weights_path is not None:
        model.load_weights(args.weights_path)
        print("Restored model parameters from {}".format(args.weights_path))

    data = ImageReader(args.data_dir, args.train_data_list, args.val_data_list,
                       (h, w), True, False)

    val_dataset = tf.data.Dataset.from_generator(
        data.val_generator, (tf.float32, tf.int32),
        (tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None])))

    val_dataset = val_dataset.batch(1)
    val_iter = iter(val_dataset)

    MeanIou = tf.metrics.MeanIoU(num_classes=num_classes)
    Accuracy = tf.metrics.Accuracy()
    mean_iou = accuracy = 0

    for step in tqdm(range(args.num_images)):

        image_batch, label_batch = next(val_iter)

        # Predictions.
        pred = model.preds(image_batch)
        seg = pred
        pred = tf.reshape(pred, [-1, ])
        gt = tf.reshape(label_batch, [-1, ])
        indices = tf.squeeze(tf.where(tf.less_equal(gt, num_classes - 1)), 1)  # ignore all labels >= num_classes
        gt = tf.cast(tf.gather(gt, indices), tf.int32)
        pred = tf.gather(pred, indices)  # Ignore all predictions for labels>=num_classes

        if args.save_figure_dir is not None:

            fig, axes = plt.subplots(1, 3, figsize=(16, 12))

            axes.flat[0].set_title('data')
            axes.flat[0].imshow((image_batch[0].numpy() + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

            axes.flat[1].set_title('mask')
            axes.flat[1].imshow(model.decode_labels(label_batch[0, :, :, 0].numpy()))

            axes.flat[2].set_title('prediction')
            axes.flat[2].imshow(model.decode_labels(seg[0, :, :, 0].numpy()))

            plt.savefig(args.save_figure_dir + '/' + str(step) + ".png")
            plt.close(fig)

        mean_iou += MeanIou(gt, pred)/ args.num_images
        accuracy += Accuracy(gt, pred) / args.num_images

    print('mean_iou: ',mean_iou)
    print('accuracy:', accuracy)

if __name__ == '__main__':

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    evaluate()