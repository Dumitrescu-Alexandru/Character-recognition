import sys
import os
import struct
import numpy as np
import tensorflow as tf
import pickle
from sources.utilities.read import read

PATH = 'C:\\Users\\alex_\\PycharmProjects\\Licenta\\EMNIST'

letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

TRAIN_IMG_DIGITS = os.path.join(PATH,'train-images-idx3-ubyte')
TRAIN_LBL_DIGITS = os.path.join(PATH,'train-labels-idx1-ubyte')
TEST_IMG_DIGITS  = os.path.join(PATH,'t10k-images-idx3-ubyte')
TEST_LBL_DIGITS  = os.path.join(PATH,'t10k-labels-idx1-ubyte')

TRAIN_IMG_LETTERS = os.path.join(PATH,'emnist-letters-train-images-idx3-ubyte')
TRAIN_LBL_LETTERS = os.path.join(PATH,'emnist-letters-train-labels-idx1-ubyte')
TEST_IMG_LETTERS  = os.path.join(PATH,'emnist-letters-test-images-idx3-ubyte')
TEST_LBL_LETTERS  = os.path.join(PATH,'emnist-letters-test-labels-idx1-ubyte')

class Data():
    nr_of_images = None

    batch_placements = [0, 0, 0]
    letter_images = []
    letter_labels = []
    letter_images_test = []
    letter_labels_test = []

    digit_images = []  # done
    digit_labels = []  # done
    digit_images_test = []  # done
    digit_labels_test = []  # done

    def __init__(self):
        digit_labels, digit_images = read(TRAIN_LBL_DIGITS, TRAIN_IMG_DIGITS)
        digit_labels_test, digit_images_test = read(TEST_LBL_DIGITS, TEST_IMG_DIGITS)

        letter_labels, letter_images = read(TRAIN_LBL_LETTERS, TRAIN_IMG_LETTERS, 'l')
        letter_labels_test, letter_images_test = read(TEST_LBL_LETTERS, TEST_IMG_LETTERS, type='l')

        self.digit_images_test = np.float32(digit_images_test)
        self.digit_labels_test = np.float32(digit_labels_test)
        self.digit_images = np.float32(digit_images)
        self.digit_labels = np.float32(digit_labels)

        self.letter_images = np.float32(letter_images)
        self.letter_labels = np.float32(letter_labels)
        self.letter_images_test = np.float32(letter_images_test)
        self.letter_labels_test = np.float32(letter_labels_test)

    def show_image(self, img_nr, type='d'):
        if type == 'd':
            image = self.digit_images[img_nr]
        else:
            image = self.letter_images[img_nr]
        from matplotlib import pyplot
        import matplotlib as mpl
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        imgplot = ax.imshow(image, cmap=mpl.cm.gray)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()

    def shuffle(self, type='d'):

        if type == 'd':
            a = self.digit_images
            b = self.digit_labels
        elif type == 'l':
            a = self.letter_images
            b = self.letter_labels
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def convert_to_onehot(self, type='d', test_or_train='train'):

        if type == 'd' and test_or_train == 'train':
            labels = self.digit_labels
        elif type == 'd' and test_or_train == 'test':
            labels = self.digit_labels_test
        elif type == 'l' and test_or_train == 'train':
            labels = self.letter_labels
        elif type == 'l' and test_or_train == 'test':
            labels = self.letter_labels_test
        else:
            labels = self.digit_labels

        one_hot = []
        if type == 'd':
            for label in labels:
                y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                y_true[int(label)] = 1
                one_hot.append(y_true)
        else:
            for label in labels:
                y_true = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                y_true[int(label - 1)] = 1
                one_hot.append(y_true)
        if type == 'd' and test_or_train == 'train':
            self.digit_labels = np.float32(one_hot)
        elif type == 'd' and test_or_train == 'test':
            self.digit_labels_test = np.float32(one_hot)
        elif type == 'l' and test_or_train == 'train':
            self.letter_labels = np.float32(one_hot)
        elif type == 'l' and test_or_train == 'test':
            self.letter_labels_test = np.float32(one_hot)

    def get_batch(self, type='d', test_or_train='train', number_of_samples=100):
        lbls = []
        img_batches = []

        if type == 'd' and test_or_train == 'train':
            if number_of_samples + self.batch_placements[0] >= 60000:
                lbls.append(self.digit_labels[self.batch_placements[0]:60000])
                img_batches.append(self.digit_images[self.batch_placements[0]:60000])
                batches_to_go = 60000 - self.batch_placements[0]
                self.shuffle('d')
                lbls.append(self.digit_labels[0:batches_to_go])
                img_batches.append(self.digit_images[0:batches_to_go])
                self.batch_placements[0] = batches_to_go
            else:
                lbls.append(self.digit_labels[self.batch_placements[0]:self.batch_placements[0] + number_of_samples])
                img_batches.append(
                    self.digit_images[self.batch_placements[0]:self.batch_placements[0] + number_of_samples])
                self.batch_placements[0] += number_of_samples
        elif type == 'l' and test_or_train == 'train':
            if number_of_samples + self.batch_placements[1] >= 124800:
                lbls.append(self.letter_labels[self.batch_placements[1]:124800])
                img_batches.append(self.letter_images[self.batch_placements[1]:124800])
                batches_to_go = 124800 - self.batch_placements[1]
                lbls.append(self.letter_labels[0:batches_to_go])
                img_batches.append(self.letter_images[0:batches_to_go])
                self.batch_placements[1] = batches_to_go
            else:
                lbls.append(self.letter_labels[self.batch_placements[1]:self.batch_placements[1] + number_of_samples])
                img_batches.append(
                    self.letter_images[self.batch_placements[1]:self.batch_placements[1] + number_of_samples])
                self.batch_placements[1] += number_of_samples
        return (lbls[0], img_batches[0])

    def combine_letters_and_digits(self):

        one_hot_train = []
        one_hot_test = []

        for label in self.digit_labels:
            c = np.zeros(36)
            c[0:10] = label
            one_hot_train.append(c)
        for label in self.letter_labels:
            c = np.zeros(36)
            c[10:36] = label
            one_hot_train.append(c)

        img_train = np.concatenate((self.digit_images, self.letter_images))
        img_test = np.concatenate((self.digit_images_test, self.letter_images_test))

        for label in self.digit_labels_test:
            c = np.zeros(36)
            c[0:10] = label
            one_hot_test.append(c)
        for label in self.letter_labels_test:
            c = np.zeros(36)
            c[10:36] = label
            one_hot_test.append(c)

        return (one_hot_train, one_hot_test, img_train, img_test)
