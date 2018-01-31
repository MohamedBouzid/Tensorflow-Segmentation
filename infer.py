import io
import os

import tensorflow as tf
import convolutional_autoencoder
from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt

if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", default="save/C7,64,2C7,64,1M2C7,64,2C7,64,1M2C7,64,2C7,64,1M2/2017-12-10_200959", type=str, help="Path to directory storing checkpointed model.")
    parser.add_argument("input_image", default="imageset-1-1.jpg", type=str, help="Path to image for which the segmentation should be performed.")
    parser.add_argument("--out", default="/tmp", type=str, help="Path to directory to store resulting image.")
    args = parser.parse_args()

    network = convolutional_autoencoder.Network()

    input_image = args.input_image
    checkpoint = args.model_dir
    out = args.out
    with tf.Session() as sess:
        saver = tf.train.Saver(tf.all_variables())
        ckpt = tf.train.get_checkpoint_state(checkpoint)
        if ckpt and ckpt.model_checkpoint_path:
            print('Restoring model: {}'.format(ckpt.model_checkpoint_path))
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            raise IOError('No model found in {}.'.format(checkpoint))


        image = np.array(ndimage.imread(input_image))  # load grayscale
        image = cv2.resize(image, (network.IMAGE_HEIGHT, network.IMAGE_WIDTH))
        image = np.multiply(image, 1.0/255)
        segmentation = sess.run(network.segmentation_result, feed_dict={
            network.inputs: np.reshape(image, [1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1])})

        segmented_image = np.dot(segmentation[0], 255)

        input_image = input_image.split('/')[-1]
        cv2.imwrite(os.path.join(args.out, input_image), segmented_image)

