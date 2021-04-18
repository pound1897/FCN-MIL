# Adapted from https://github.com/DrSleep/tensorflow-deeplab-lfov
from PIL import Image
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from fcn_mil import MIL_FCN
import numpy as np
import matplotlib.pyplot as plt


IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

def read_labeled_image_list(data_dir, data_list):
    """Reads txt file containing paths to images and ground truth masks.

    Args:
        data_dir: path to the directory with images and masks.
        data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
    Returns:
        Two lists with all file names for images and masks, respectively.
    """
    f = open(data_list, 'r')
    images = []
    masks = []
    for line in f:
        image, mask = line.strip("\n").split(' ')
        images.append(data_dir + image)
        masks.append(data_dir + mask)
    #    images, val_images, masks, val_labels = model_selection.train_test_split(images, masks, test_size=1449, train_size=10582)
    #    return images,  val_images ,masks , val_labels
    return images, masks

def read_images(image_filename, label_filename, input_size, random_scale):
    """Read one image and its corresponding mask with optional pre-processing.

    Args:
        img_filename: a string with image filename.
        label_filename: a string with label image filename.
        input_size: a tuple with (height, width) values.
                  If not given, return images of original size.
        random_scale: whether to randomly scale the images prior
                    to random crop.
    Returns:
        Two tensors: the decoded image and its mask.
    """
    img_contents = tf.io.read_file(image_filename)
    label_contents = tf.io.read_file(label_filename)

    img = tf.image.decode_jpeg(img_contents, channels=3)
    label = tf.image.decode_png(label_contents, channels=1)

    if input_size is not None:
        h, w = input_size
        if random_scale:
            scale = tf.random.uniform([1], minval=0.75, maxval=1.25, dtype=tf.float32, seed=None)
            h_new = tf.cast(tf.multiply(tf.cast(tf.shape(img)[0], dtype=tf.float32), scale), dtype=tf.int32)
            w_new = tf.cast(tf.multiply(tf.cast(tf.shape(img)[1], dtype=tf.float32), scale), dtype=tf.int32)
            new_shape = tf.squeeze(tf.stack([h_new, w_new]), axis=1)
            img = tf.image.resize(img, new_shape)
            label = tf.image.resize(tf.expand_dims(label, 0), new_shape, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
            label = tf.squeeze(label, axis=0)  # resize_image_with_crop_or_pad accepts 3D-tensor.

        img = tf.image.resize_with_crop_or_pad(img, h, w)
        label = tf.image.resize_with_crop_or_pad(label, h, w)
    # RGB -> BGR.
    img_r, img_g, img_b = tf.split(axis=2, num_or_size_splits=3, value=img)
    img = tf.cast(tf.concat([img_b, img_g, img_r], 2), dtype=tf.float32)
    # Extract mean.
    img -= IMG_MEAN
    return img, label

def data_generator(data_dir, data_list, input_size, random_scale):
    """Get the data generator.

    Args:
        data_dir: path to the directory with images and masks.
        data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
        input_size: a tuple with (height, width) values.
                    If not given, return images of original size.
        random_scale: whether to randomly scale the images prior
                        to random crop.
    """
    images, masks = read_labeled_image_list(data_dir, data_list)
    for image_filename, label_filename in zip(images, masks):
        image, label = read_images(image_filename, label_filename, input_size, random_scale)
        yield tf.convert_to_tensor(image), tf.convert_to_tensor(label)

class ImageReader(object):
    """Generic ImageReader which reads images and corresponding segmentation
       masks from the disk.
    """

    def __init__(self, data_dir, train_data_list, val_data_list, input_size, random_scale, is_shuffle):
        """Initialise an ImageReader.

        Args:
          data_dir: path to the directory with images and masks.
          data_list: path to the file with lines of the form '/path/to/image /path/to/mask'.
          input_size: a tuple with (height, width) values, to which all the images will be resized.
          random_scale: whether to randomly scale the images prior to random crop.
          is_shuffle: whether to shuffle the training dataset.
        """
        self.data_dir = data_dir
        self.train_data_list = train_data_list
        self.val_data_list = val_data_list
        self.input_size = input_size
        self.random_scale = random_scale
        self.is_shuffle = is_shuffle

    def train_generator(self):
        images, masks = read_labeled_image_list(self.data_dir,self.train_data_list)
        if self.is_shuffle:
            sort = np.arange(len(images))
            np.random.shuffle(sort)
            images = [images[i] for i in sort]
            masks = [masks[i] for i in sort]
        for image_filename, label_filename in zip(images, masks):
            image, label = read_images(image_filename, label_filename, self.input_size,self.random_scale)
            yield tf.convert_to_tensor(image), tf.convert_to_tensor(label)

    def val_generator(self):
        images, masks = read_labeled_image_list(self.data_dir, self.val_data_list)
        for image_filename, label_filename in zip(images, masks):
            image, label = read_images(image_filename, label_filename, self.input_size, self.random_scale)
            yield tf.convert_to_tensor(image), tf.convert_to_tensor(label)





