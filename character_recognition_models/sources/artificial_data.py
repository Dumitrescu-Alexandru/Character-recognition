
import os
import numpy as np
import pickle
from sources.data import Data

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


class ArtificialData(Data):
    file_name = 'data-artificiale-ubyte'
    datele_mele = Data()
    order = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
             14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: [],
             27: [], 28: [], 29: [], 30: [], 31: [], 32: [], 33: [], 34: [], 35: [], 36: []}

    def __init__(self):

        super().__init__()
        self.show_image(img_nr=0, type='d')

    def show_some_data(self):

        print("\n\n\n", np.shape(self.digit_images))
        print(np.shape(self.letter_images))
        for j in range(5):
            self.show_image(j * 10000, 'd')
            self.datele_mele.show_image(j * 10000, 'd')

    def show_image_order(self, image):
        from matplotlib import pyplot
        import matplotlib as mpl
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        imgplot = ax.imshow(image, cmap=mpl.cm.gray)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()

    def get_in_order(self):
        for i in range(60000):
            self.order[self.datele_mele.digit_labels[i]].append(self.datele_mele.digit_images[i])
        for i in range(124800):
            self.order[int(self.datele_mele.letter_labels[i]) + 9].append(self.datele_mele.letter_images[i])

    def generate_image(self, string):

        l = 0
        new_image = [[None] * len(string) * 28 * 28]

        new_image = np.float32(new_image)

        new_image = np.array(new_image)

        new_image = new_image.reshape(28, (28 * len(string)))

        for i in string:

            if (ord(i) > 96 and ord(i) < 123):
                z = ord(i)
                z = z - 87
                c = np.shape(self.order[z])
                max = c[0] - 1
                nr = np.random.randint(0, max - 1)
                new_image[0:28, l * 28:(l + 1) * 28] = self.order[z][nr]
                l += 1
            else:
                z = ord(i)
                z = z - 48
                c = np.shape(self.order[z])
                max = c[0] - 1
                nr = np.random.randint(0, max - 1)
                new_image[0:28, l * 28:(l + 1) * 28] = self.order[z][nr]
                l += 1

        return new_image

    def generate_string(self, size):

        string = ""

        a = np.random.randint(0, 36, size)
        for b in a:
            if b >= 10:
                string += chr(b + 87)
            else:
                string += chr(48 + b)
        return string

    def add_images(self, images, created=False, file_name='test.bin'):

        created = os.path.isfile(file_name)
        if created == False:
            self.create_file(file_name)
            file = open(file_name, 'wb')
        else:
            file = open(file_name, 'ab')
        pickle.dump(images, file)
        file.close()

    def add_labels(self, labels, lbls_file_name='lbl_test.bin'):
        created_lbls = os.path.isfile(lbls_file_name)
        if created_lbls == False:
            self.create_file(lbls_file_name)
            file_lbls = open(lbls_file_name, 'wb')
        else:
            file_lbls = open(lbls_file_name, 'ab')
        pickle.dump(labels, file_lbls)
        file_lbls.close()

    def create_file(self, name='test.bin'):
        file = open(name, 'wb')
        file.close()

    def read_images(self, name='test.bin'):
        fisier_imagini = open(name, 'rb')
        images = []
        while 1:
            try:
                images.append(pickle.load(fisier_imagini))
            except EOFError:
                break
        fisier_imagini.close()
        return images

    def read_labels(self, name='lbl_test.bin'):
        fisier_labels = open(name, 'rb')
        labels = []
        while 1:
            try:
                labels.append(pickle.load(fisier_labels))
            except EOFError:
                break
        fisier_labels.close()
        return labels