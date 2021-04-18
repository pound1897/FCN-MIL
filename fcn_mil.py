from tensorflow.keras.models import Sequential,Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import *
import tensorflow as tf
import numpy as np
from PIL import Image

weight_decay = 5e-4
class MIL_FCN(Model):

    def __init__(self, num_classes, batch_size, optimizer):

        # Shape for input x: (B, W, H, C)
        # Shape for input labels : (B, num_classes)

        super(MIL_FCN, self).__init__()

        self.num_classes = num_classes
        self.batch_size = batch_size
        self.optimizer = optimizer

        self.vgg16 = Sequential([
            Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1'),
            Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2'),
            MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool'),

            # Block 2
            Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1'),
            Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2'),
            MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool'),

            # Block 3
            Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1'),
            Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2'),
            Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3'),
            MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool'),

            # Block 4
            Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1'),
            Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2'),
            Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3'),
            MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool'),

            # Block 5
            Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1'),
            Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2'),
            Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3'),
            MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')

        ])

        self.heatmap = Sequential([
            Conv2D(num_classes, (3, 3), activation='relu', kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(weight_decay), padding='same')
            ])

    def call(self, inputs, training=None):

        vgg = self.vgg16(inputs)
        heatmap = self.heatmap(vgg)

        return heatmap

    def prepare_label(self, input_batch, new_size):
        """Resize masks and perform one-hot encoding.

        Args:
            input_batch: input tensor of shape [batch_size H W 1].
            new_size: a tensor with new height and width.

        Returns:
            Outputs a tensor of shape [batch_size h w 21]
            with last dimension comprised of 0's and 1's only.
        """
        input_batch = tf.image.resize(input_batch, new_size,
                                      method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)  # As labels are integer numbers, need to use NN interp.
        input_batch = tf.squeeze(input_batch, axis=[3])  # Reducing the channel dimension.
        input_batch = tf.one_hot(input_batch, depth=self.num_classes)

        return input_batch

    def decode_labels(self, mask):
        """Decode batch of segmentation masks.

        Args:
          label_batch: result of inference after taking argmax.

        Returns:
          An batch of RGB images of the same size
        """

        # Colour map.
        label_colours = [(0, 0, 0)
                         # 0=background
            , (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128)
                         # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
            , (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0)
                         # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
            , (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128)
                         # 11=diningtable, 12=dog, 13=horse, 14=motorbike, 15=person
            , (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)]
        # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor

        img = Image.new('RGB', (len(mask[0]), len(mask)))
        pixels = img.load()
        for j_, j in enumerate(mask):
            for k_, k in enumerate(j):
                if k < self.num_classes:
                    pixels[k_, j_] = label_colours[k]

        return np.array(img)

    def preds(self, img_batch):
        """Create the network and run inference on the input batch.

        Args:
          input_batch: batch of pre-processed images.

        Returns:
          Argmax over the predictions of the network of the same shape as the input.
        """
        heatmap = self.call(img_batch)
        heatmap = tf.image.resize(heatmap, tf.shape(img_batch)[1:3, ])  # Upsample the raw output bilinearly to match the size of the input
        heatmap = tf.argmax(heatmap, axis=3)
        heatmap = tf.expand_dims(heatmap, axis=3)  # Create 4D-tensor.

        return tf.cast(heatmap, tf.uint8)

    def mil_loss(self, heatmap, label_batch, new_shape):

        heatmap = tf.image.resize(heatmap, new_shape)
        probabilities = tf.nn.softmax(heatmap)
        prediction = tf.reshape(probabilities, [self.batch_size, -1, self.num_classes])
        mil_loss = 0
        for label_number in range(self.batch_size):
            label = label_batch[label_number]
            label = tf.reshape(label, [-1])
            label = tf.cast(label, dtype=tf.int32)  # Convert label to int32
            cond = tf.equal(label, tf.constant(255))
            new_label = tf.where(cond, tf.zeros_like(label),
                                 label)  # Replace the void label(255) by the background label(0)
            weak_label, _ = tf.unique(new_label)  # Summarise the labels by taking the set out of the list of labels --discarding repeated elements

            loss_per_image_vector = tf.map_fn(lambda i: tf.math.log(tf.reduce_max(prediction[label_number, :, i])),
                                              weak_label, dtype=tf.float32)
            loss_per_image = tf.reduce_sum(loss_per_image_vector)
            loss_per_image = -tf.divide(loss_per_image, tf.cast(tf.size(weak_label), dtype=tf.float32))
            mil_loss = mil_loss + loss_per_image

        mil_loss = tf.divide(mil_loss, self.batch_size)  # Averaged loss for the batch

        return mil_loss

    def gradientDescent(self, img_batch, label_batch):

        with tf.GradientTape() as tape:
            heatmap = self.call(img_batch)
            loss_mil = self.mil_loss(heatmap, label_batch, tf.shape(img_batch)[1:3, ])
        grads = tape.gradient(loss_mil, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))

        return loss_mil





