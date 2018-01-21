import math
import os
import time
from math import ceil

import cv2
import matplotlib
from scipy import ndimage
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import gen_nn_ops

from conv2d import Conv2d
from max_pool_2d import MaxPool2d
import datetime
import io

np.set_printoptions(threshold=np.nan)


# @ops.RegisterGradient("MaxPoolWithArgmax")
# def _MaxPoolWithArgmaxGrad(op, grad, unused_argmax_grad):
#     return gen_nn_ops._max_pool_grad(op.inputs[0],
#                                      op.outputs[0],
#                                      grad,
#                                      op.get_attr("ksize"),
#                                      op.get_attr("strides"),
#                                      padding=op.get_attr("padding"),
#                                      data_format='NHWC')


class Network:
    IMAGE_HEIGHT = 192
    IMAGE_WIDTH  = 192
    IMAGE_CHANNELS = 1

    def __init__(self, layers=None, per_image_standardization=True, batch_norm=True, skip_connections=True):
        # Define network - ENCODER (decoder will be symmetric).
        s=3
        if layers == None:
            layers = []
            layers.append(Conv2d(kernel_size=s, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_1')) # 61 x 61 x 64
            layers.append(Conv2d(kernel_size=s, strides=[1, 1, 1, 1], output_channels=64, name='conv_1_2')) # 55 x 55 x 64
            layers.append(MaxPool2d(kernel_size=2, name='max_1', skip_connection=skip_connections)) 				# 54 x 54

            layers.append(Conv2d(kernel_size=s, strides=[1, 1, 1, 1], output_channels=64, name='conv_2_1')) # 24 x 24 x 64 
            layers.append(Conv2d(kernel_size=s, strides=[1, 1, 1, 1], output_channels=64, name='conv_2_2')) # 18
            layers.append(MaxPool2d(kernel_size=2, name='max_2', skip_connection=skip_connections))  				# 17

            layers.append(Conv2d(kernel_size=s, strides=[1, 1, 1, 1], output_channels=64, name='conv_3_1')) # 24 x 24 x 64 
            layers.append(Conv2d(kernel_size=s, strides=[1, 1, 1, 1], output_channels=64, name='conv_3_2')) # 18
            layers.append(MaxPool2d(kernel_size=2, name='max_3', skip_connection=skip_connections))  				# 17 

            layers.append(Conv2d(kernel_size=s, strides=[1, 1, 1, 1], output_channels=64, name='conv_4_1')) # 6
            layers.append(Conv2d(kernel_size=s, strides=[1, 1, 1, 1], output_channels=64, name='conv_4_2')) 
            layers.append(MaxPool2d(kernel_size=2, name='max_4', skip_connection=skip_connections))

            layers.append(Conv2d(kernel_size=s, strides=[1, 1, 1, 1], output_channels=64, name='conv_5_1')) # 6
            layers.append(Conv2d(kernel_size=s, strides=[1, 1, 1, 1], output_channels=64, name='conv_5_2')) 
            layers.append(MaxPool2d(kernel_size=2, name='max_5'))

        self.inputs = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, self.IMAGE_CHANNELS],
                                     name='inputs')
        self.targets = tf.placeholder(tf.float32, [None, self.IMAGE_HEIGHT, self.IMAGE_WIDTH, 1], name='targets')
        self.is_training = tf.placeholder_with_default(False, [], name='is_training')
        self.description = ""

        self.layers = {}

        if per_image_standardization:
            list_of_images_norm = tf.map_fn(tf.image.per_image_standardization, self.inputs)
            net = tf.stack(list_of_images_norm)
        else:
            net = self.inputs

        # ENCODER
        for layer in layers:
            self.layers[layer.name] = net = layer.create_layer(net)
            self.description += "{}".format(layer.get_description())

        print("Current input shape: ", net.get_shape())

        layers.reverse()
        Conv2d.reverse_global_variables()

        # DECODER
        for layer in layers:
            net = layer.create_layer_reversed(net, prev_layer=self.layers[layer.name])

        self.segmentation_result = tf.sigmoid(net)

        # segmentation_as_classes = tf.reshape(self.y, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH, 1])
        # targets_as_classes = tf.reshape(self.targets, [50 * self.IMAGE_HEIGHT * self.IMAGE_WIDTH])
        # print(self.y.get_shape())
        # self.cost = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(segmentation_as_classes, targets_as_classes))
        print('segmentation_result.shape: {}, targets.shape: {}'.format(self.segmentation_result.get_shape(),
                                                                        self.targets.get_shape()))

        # MSE loss

        self.cost = tf.sqrt(tf.reduce_mean(tf.square(self.segmentation_result - self.targets)))

        self.train_op = tf.train.AdamOptimizer().minimize(self.cost)
        with tf.name_scope('accuracy'):
            argmax_probs = tf.round(self.segmentation_result)  # 0x1
            correct_pred = tf.cast(tf.equal(argmax_probs, self.targets), tf.float32)
            self.accuracy = tf.reduce_mean(correct_pred)

            tf.summary.scalar('accuracy', self.accuracy)

        self.summaries = tf.summary.merge_all()


class Dataset:
    def __init__(self, batch_size, folder):
        self.batch_size = batch_size

        train_files, validation_files, test_files = self.train_valid_test_split(
            os.listdir(os.path.join(folder, 'inputs')))

        self.train_inputs, self.train_targets = self.file_paths_to_images(folder, train_files)
        self.test_inputs, self.test_targets = self.file_paths_to_images(folder, test_files, True)

        self.pointer = 0

    def file_paths_to_images(self, folder, files_list, verbose=False):
        inputs = []
        targets = []

        for file in files_list:
            input_image = os.path.join(folder, 'inputs', file)
            target_image = os.path.join(folder, 'targets', file)

            test_image = np.array(ndimage.imread(input_image))  # load grayscale
            # test_image = np.multiply(test_image, 1.0 / 255)
            inputs.append(test_image)

            target_image = ndimage.imread(target_image)
            target_image = cv2.threshold(target_image, 127, 1, cv2.THRESH_BINARY)[1]
            targets.append(target_image)

        return inputs, targets

    def train_valid_test_split(self, X, ratio=None):
        if ratio is None:
            ratio = (0.7, .15, .15)

        N = len(X)
        return (
            X[:int(ceil(N * ratio[0]))],
            X[int(ceil(N * ratio[0])): int(ceil(N * ratio[0] + N * ratio[1]))],
            X[int(ceil(N * ratio[0] + N * ratio[1])):]
        )

    def num_batches_in_epoch(self):
        return int(math.floor(len(self.train_inputs) / self.batch_size))

    def reset_batch_pointer(self):
        permutation = np.random.permutation(len(self.train_inputs))
        self.train_inputs = [self.train_inputs[i] for i in permutation]
        self.train_targets = [self.train_targets[i] for i in permutation]

        self.pointer = 0

    def next_batch(self):
        inputs = []
        targets = []
        # print(self.batch_size, self.pointer, self.train_inputs.shape, self.train_targets.shape)
        for i in range(self.batch_size):
            inputs.append(np.array(self.train_inputs[self.pointer + i]))
            targets.append(np.array(self.train_targets[self.pointer + i]))

        self.pointer += self.batch_size
				
        return np.array(inputs, dtype=np.uint8), np.array(targets, dtype=np.uint8)

    @property
    def test_set(self):
        return np.array(self.test_inputs, dtype=np.uint8), np.array(self.test_targets, dtype=np.uint8)


def draw_results(test_inputs, test_targets, test_segmentation, test_accuracy, network, batch_num):
    n_examples_to_plot = 12
    fig, axs = plt.subplots(4, n_examples_to_plot, figsize=(n_examples_to_plot * 3, 10))
    fig.suptitle("Accuracy: {}, {}".format(test_accuracy, network.description), fontsize=20)
    for example_i in range(n_examples_to_plot):
        axs[0][example_i].imshow(test_inputs[example_i], cmap='gray')
        axs[1][example_i].imshow(test_targets[example_i].astype(np.float32), cmap='gray')
        axs[2][example_i].imshow(
            np.reshape(test_segmentation[example_i], [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]),
            cmap='gray')

        test_image_thresholded = np.array(
            [0 if x < 0.5 else 255 for x in test_segmentation[example_i].flatten()])
        axs[3][example_i].imshow(
            np.reshape(test_image_thresholded, [network.IMAGE_HEIGHT, network.IMAGE_WIDTH]),
            cmap='gray')

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)

    IMAGE_PLOT_DIR = 'image_plots/'
    if not os.path.exists(IMAGE_PLOT_DIR):
        os.makedirs(IMAGE_PLOT_DIR)

    plt.savefig('{}/figure{}.jpg'.format(IMAGE_PLOT_DIR, batch_num))
    return buf


def train():
    BATCH_SIZE = 10

    network = Network()

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

    # create directory for saving models
    os.makedirs(os.path.join('save', network.description, timestamp))

    dataset = Dataset(BATCH_SIZE, 'data192_192')

    inputs, targets = dataset.next_batch()



    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter('{}/{}-{}'.format('logs', network.description, timestamp),
                                               graph=tf.get_default_graph())
        saver = tf.train.Saver(tf.all_variables(), max_to_keep=None)

        test_accuracies = []
        # Fit all training data
        n_epochs = 50
        global_start = time.time()
        for epoch_i in range(n_epochs):
            dataset.reset_batch_pointer()

            for batch_i in range(dataset.num_batches_in_epoch()):
                batch_num = epoch_i * dataset.num_batches_in_epoch() + batch_i + 1

                start = time.time()
                batch_inputs, batch_targets = dataset.next_batch()
                batch_inputs = np.reshape(batch_inputs,
                                          (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                batch_targets = np.reshape(batch_targets,
                                           (dataset.batch_size, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))

                batch_inputs = np.multiply(batch_inputs, 1.0 / 255)


                cost, _ = sess.run([network.cost, network.train_op],
                                   feed_dict={network.inputs: batch_inputs, network.targets: batch_targets,
                                              network.is_training: True})
                print("cost.shape  = ",cost.shape)
                end = time.time()
                print('{}/{}, epoch: {}, cost: {}, batch time: {}'.format(batch_num,
                                                                          n_epochs * dataset.num_batches_in_epoch(),
                                                                          epoch_i, cost, end - start))

                if batch_num % BATCH_SIZE == 0 or batch_num == n_epochs * dataset.num_batches_in_epoch():
                    test_inputs, test_targets = dataset.test_set
                    # test_inputs, test_targets = test_inputs[:100], test_targets[:100]

                    test_inputs = np.reshape(test_inputs, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                    test_targets = np.reshape(test_targets, (-1, network.IMAGE_HEIGHT, network.IMAGE_WIDTH, 1))
                    test_inputs = np.multiply(test_inputs, 1.0 / 255)


                    summary, test_accuracy = sess.run([network.summaries, network.accuracy],
                                                      feed_dict={network.inputs: test_inputs,
                                                                 network.targets: test_targets,
                                                                 network.is_training: False})

                    summary_writer.add_summary(summary, batch_num)

                    print('Step {}, test accuracy: {}'.format(batch_num, test_accuracy))
                    test_accuracies.append((test_accuracy, batch_num))
                    print("Accuracies in time: ", [test_accuracies[x][0] for x in range(len(test_accuracies))])
                    max_acc = max(test_accuracies)
                    print("Best accuracy: {} in batch {}".format(max_acc[0], max_acc[1]))
                    print("Total time: {}".format(time.time() - global_start))


                    if test_accuracy >= max_acc[0]:
                        checkpoint_path = os.path.join('save', network.description, timestamp, 'model.ckpt')
                        saver.save(sess, checkpoint_path, global_step=batch_num)


if __name__ == '__main__':
    train()
