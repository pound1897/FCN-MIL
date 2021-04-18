"""Evaluation script for the DeepLab-LargeFOV network on the validation subset
   of PASCAL VOC dataset.

This script evaluates the model on around 1500 validation images.
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import time
import argparse
import shutil
import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from DataPreprocess import ImageReader,IMG_MEAN
from fcn_mil import MIL_FCN

random_scale =True
is_shuffle = True

# model parameters
input_size = '256,256'
batch_size = 32
num_classes = 21

# train datasets and validation datasets
data_directory = 'datasets\VOCdevkit'
train_data_list_path = './dataset/train.txt'
val_data_list_path = './dataset/val.txt'

# optimizer parameters
learning_rate = 1e-5
momentum = 0.9
weight_decay = 5e-4

# train and validate epochs
train_epochs = 5000
val_epochs = 1400//batch_size

save_figure_dir = './train_images'
save_num_images = 3
save_pred_every = 100

save_model_dir = './MIL-FCN'
weights_path   = None
summaries_dir = './summaries'

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
    parser.add_argument("--train_epochs", type=int, default=train_epochs,
                        help="Number of training epochs.")
    parser.add_argument("--val_epochs", type=int, default=val_epochs,
                        help="Number of validating epochs.")
    parser.add_argument("--save_model_dir", type=str, default=save_model_dir,
                        help="Where to save model parameters .")
    parser.add_argument("--save_figure_dir", type=str, default=save_figure_dir,
                        help="Where to save figures with predictions.")
    parser.add_argument("--save_num_images", type=int, default=save_num_images,
                        help="How many images to save.")
    parser.add_argument("--save_pred_every", type=int, default=save_pred_every,
                        help="Save figure with predictions and ground truth every often.")
    parser.add_argument("--weights_path", type=str, default=weights_path,
                        help="Path to the file with model weights. "
                             "If not set, all the variables are initialised randomly.")
    parser.add_argument("--summaries_dir", type=str, default=summaries_dir,
                        help="Path to the file where variables are saved for TensorBoard.")
    return parser.parse_args()

def train():

    """Create the model and start the training."""
    args = get_arguments()
    h, w = map(int, args.input_size.split(','))

    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learning_rate, beta_1=args.momentum,
                                         beta_2=0.996, decay=args.weight_decay)

    model = MIL_FCN(args.num_classes, args.batch_size, optimizer)
    model.build((None, h, w, 3))
    if args.weights_path is not None:
        model.vgg16.load_weights(args.weights_path)
        print("Restored model parameters from {}".format(args.weights_path))

    data = ImageReader(args.data_dir, args.train_data_list, args.val_data_list,
                       (h, w), random_scale, is_shuffle)
    train_dataset = tf.data.Dataset.from_generator(
        data.train_generator, (tf.float32, tf.int32),
        (tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None])))

    train_dataset = train_dataset.repeat()
    train_dataset = train_dataset.batch(args.batch_size)
    train_iter = iter(train_dataset)

    starting_time = time.asctime(time.localtime())
    if os.path.exists('time_logs.txt'):
        os.remove('time_logs.txt')
    with open('time_logs.txt', 'w+') as f:
        print("Training started on: ", starting_time, file=f)

    if args.save_figure_dir is not None:
        if os.path.exists(args.save_figure_dir):
            shutil.rmtree(args.save_figure_dir)
            os.mkdir(args.save_figure_dir)
        else :
            os.mkdir(args.save_figure_dir)

    if args.save_model_dir is not None:
        if os.path.exists(args.save_model_dir):
            shutil.rmtree(args.save_model_dir)
            os.mkdir(args.save_model_dir)
        else :
            os.mkdir(args.save_model_dir)

    if os.path.exists(args.summaries_dir):
        shutil.rmtree(args.summaries_dir)
    else :
        os.makedirs(args.summaries_dir)

    summary_writer = tf.summary.create_file_writer(args.summaries_dir)
    # Iterate over training epochs.
    for epoch in range(args.train_epochs + 1):

        img_batch, label_batch = next(train_iter)

        start_time = time.time()
        train_loss = model.gradientDescent(img_batch, label_batch)

        print('epoch: {} '.format(epoch), 'train_loss: {:.5f}'.format(train_loss),
              'time: {:.3f}s'.format(time.time() - start_time))

        with summary_writer.as_default():
            tf.summary.scalar('train_loss', train_loss, step=epoch)

        if epoch % args.save_pred_every == 0:

            train_time = time.asctime(time.localtime())
            with open('time_logs.txt', 'a') as f:
                print('train_loss: {:.5f}'.format(train_loss), 'train in: ', train_time, file=f)

            val_dataset = tf.data.Dataset.from_generator(
                data.val_generator, (tf.float32, tf.int32),
                (tf.TensorShape([None, None, None]), tf.TensorShape([None, None, None])))

            val_dataset = val_dataset.repeat()
            val_dataset = val_dataset.batch(args.batch_size)
            val_iter = iter(val_dataset)
            val_loss = 0

            for i in range(args.val_epochs):

                val_img_batch, val_label_batch = next(val_iter)
                heatmap = model(val_img_batch)
                loss_temp = model.mil_loss(heatmap, val_label_batch, tf.shape(val_img_batch)[1:3, ])
                val_loss += loss_temp

                if i == 0:
                    pred = model.preds(val_img_batch)
                    fig, axes = plt.subplots(args.save_num_images, 3, figsize=(16, 12))
                    for j in range(args.save_num_images):
                        axes.flat[j * 3].set_title('data')
                        axes.flat[j * 3].imshow(
                            (val_img_batch[j, :, :, :].numpy() + IMG_MEAN)[:, :, ::-1].astype(np.uint8))

                        axes.flat[j * 3 + 1].set_title('mask')
                        axes.flat[j * 3 + 1].imshow(model.decode_labels(val_label_batch[j, :, :, 0].numpy()))

                        axes.flat[j * 3 + 2].set_title('prediction')
                        axes.flat[j * 3 + 2].imshow(model.decode_labels(pred[j, :, :, 0].numpy()))

                    plt.savefig(args.save_figure_dir + '/' + str(epoch) + ".png")
                    plt.close(fig)
                    if args.save_model_dir is not None:
                        if not os.path.exists(args.save_model_dir):
                            os.makedirs(args.save_model_dir)
                        model_name = 'MIL_FCN' + '_' + str(epoch) + '.h5'
                        checkpoint_path = os.path.join(args.save_model_dir, model_name)
                        model.save_weights(checkpoint_path, save_format='h5')
                        print('The checkpoint has been created.')

            val_loss = val_loss / args.val_epochs
            print('epoch: {} '.format(epoch), 'val_loss: {:.5f}'.format(val_loss),
                  'time: {:.3f}s'.format(time.time() - start_time))
            validation_time = time.asctime(time.localtime())
            with open('time_logs.txt', 'a') as f:
                print('val_loss: {:.5f}'.format(val_loss), 'validate in: ', validation_time, file=f)
            with summary_writer.as_default():
                tf.summary.scalar('val_loss', val_loss, step=epoch)

    end_time = time.asctime(time.localtime())
    with open('time_logs.txt', 'a') as f:
        print("Training ended on: ", end_time, file=f)

if __name__ == '__main__':

    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession

    config = ConfigProto()
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    train()












