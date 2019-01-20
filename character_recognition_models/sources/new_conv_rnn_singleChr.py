from sources.data import Data
import numpy as np
from cuvinte.word_generator import data_cleaning
import pickle
import os
import cv2
import tensorflow as tf
from sources.utilities.convo_functions import init_weights
from sources.utilities.convo_functions import init_bias
from sources.utilities.convo_functions import conv2d
from sources.utilities.convo_functions import max_pool_2by2
from sources.utilities.convo_functions import convolutional_layer
from sources.utilities.convo_functions import normal_full_layer
PATH_TO_BINARIES = "C:\\Users\\alex_\\PycharmProjects\\Licenta\\LSTM\\binaries\\"
SLICE_DIM = 100
class CONV_LSTM(Data):

    smaller_oh = []
    smaller_labels =[]
    smaller_images = []
    small_word_number = []

    small_batch_placement = 0
    batch_placement = 0
    lstm_data_labels_oh = []
    lstm_data_labels = []
    lstm_data_images = []
    words = []
    words_training_number = None        #number of ARtfi. data red
    train_letter_number = None
    letters_train= []
    letters_test = []
    letters_train_labels = []
    letters_test_labels = []
    upper_lower_case = []
    order_upper = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
             14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: []}
    order_lower = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: [], 7: [], 8: [], 9: [], 10: [], 11: [], 12: [], 13: [],
             14: [], 15: [], 16: [], 17: [], 18: [], 19: [], 20: [], 21: [], 22: [], 23: [], 24: [], 25: [], 26: []}

    def __init__(self):
        super().__init__()
        words = data_cleaning()
        self.words = [item.rstrip() for item in words]
        self.letters_train = self.letter_images
        self.letters_test = self.letters_test
        self.letters_train_labels = self.letter_labels
        self.letters_test_labels = self.letter_labels_test
        file = open(r"../letsarrange.txt")
        self.train_letter_number = int(file.read())
        self.train_letter_number = int(self.train_letter_number)
        self.train_letter_number = self.train_letter_number - 60000
        '''
            THIS WILL BE TRAINED ON THE self.train_letter_number,
            THE NUMBER TO WHICH I HAVE GOT UP TO THIS POINT
        '''
        self.letters_train = self.letters_train[:self.train_letter_number]
        self.letters_train_labels = self.letters_train_labels[:self.train_letter_number]

        file.close()
        file = open(r"../aranjate.txt")
        aux =  file.read()
        self.upper_lower_case = aux.split()
        self.upper_lower_case = list(map(int,self.upper_lower_case))
        file.close()

    def get_in_order(self):
        for i in range(self.train_letter_number):
            if self.upper_lower_case[i] == 1:
                self.order_upper[int(self.letter_labels[i]) - 1].append(self.letters_train[i])
            else:
                self.order_lower[int(self.letter_labels[i]) - 1].append(self.letters_train[i])

    def create_1_chr_data(self,file_name_image="image_letters.bin",file_name_lbls="label_letters.bin",file_name_oh="letter_oh.bin"):
        path = "one_letter_lower//"
        imgs = []
        lbls = []
        lbls_oh = []
        for key, chrs in self.order_lower.items():
            for chr in chrs:
                imgs.append(chr)
                lbls.append(str(key+97))
                lbl = np.zeros(27)
                lbl[key] = 1
                lbls_oh.append(lbl)

        file = open(path + file_name_image, 'ab')
        pickle.dump(imgs, file)
        file.close()
        file = open(path + file_name_lbls, 'ab')
        pickle.dump(lbls, file)
        file.close()
        file = open(path + file_name_oh, 'ab')
        pickle.dump(lbls_oh, file)
        file.close()

    def append_data(self):
        path = "one_letter_lower//"
        lbl_file_name = path + "image_letters.bin"
        img_file_name = path + "label_letters.bin"
        oh_labels = path + "letter_oh.bin"

        images_file = open(img_file_name, 'rb')
        images = []
        while 1:
            try:
                images.append(pickle.load(images_file))
            except EOFError:
                break
        images_file.close()
        imgs = []
        for img in images:
            imgs = imgs + img

        label_word_file = open(lbl_file_name, 'rb')
        labels = []
        while 1:
            try:
                labels.append(pickle.load(label_word_file))
            except EOFError:
                break
        label_word_file.close()
        lbls = []
        for lbl in labels:
            lbls = lbls + lbl

        label_oh_file = open(oh_labels, 'rb')
        labels_oh = []
        while 1:
            try:
                labels_oh.append(pickle.load(label_oh_file))
            except EOFError:
                break
        label_word_file.close()
        lbls_oh = []
        for lbl_oh in labels_oh:
            lbls_oh = lbls_oh + lbl_oh
        self.small_word_number = np.shape(lbls)[0]
        self.smaller_images = lbls
        self.smaller_labels = imgs
        self.smaller_oh = lbls_oh

    def show_image_lettes(self,image):

        from matplotlib import pyplot
        import matplotlib as mpl
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        imgplot = ax.imshow(image, cmap=mpl.cm.gray)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()

    def shuffle_small_lsmt(self):
        a = self.smaller_images
        b = self.smaller_labels
        c = self.smaller_oh

        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def get_lstm_batch(self, test_or_train='train', number_of_samples=100):
        lbls = []
        img_batches = []

        if number_of_samples + self.batch_placement >= self.small_word_number:
            lbls.append(self.smaller_oh[self.batch_placement:self.small_word_number])
            img_batches.append(self.smaller_images[self.batch_placement:self.small_word_number])
            batches_to_go = self.small_word_number - self.batch_placement
            self.shuffle_small_lsmt()
            lbls.append(self.smaller_oh[0:batches_to_go])
            img_batches.append(self.smaller_images[0:batches_to_go])
            self.batch_placement = batches_to_go
        else:
            lbls.append(self.smaller_oh[self.batch_placement:self.batch_placement + number_of_samples])
            img_batches.append(self.smaller_images[self.batch_placement:self.batch_placement + number_of_samples])
            self.batch_placement += number_of_samples
        return (lbls[0], img_batches[0])
    '''self.small_word_number = np.shape(lbls)[0]
        self.smaller_images = lbls
        self.smaller_labels = imgs
        self.smaller_oh = lbls_oh'''

    def model(self,file_name="conv_lstm.ckpt"):
        x = tf.placeholder(tf.float32, shape=[None, 28, 28])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        y_true = tf.placeholder(tf.float32, shape=[None, 27])

        convo_1 = convolutional_layer(x_image, shape=[3, 3, 1, 32])
        convo_1_pooling = max_pool_2by2(convo_1)
        convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 32, 64])
        convo_2_pooling = max_pool_2by2(convo_2)
        to_be_fed = convo_2_pooling
        print(np.shape(convo_2_pooling))
        to_be_fed = tf.reshape(to_be_fed, shape=[-1,1, 3136])
        cell_1 = tf.contrib.rnn.BasicLSTMCell(3136)
        cell_2 = tf.contrib.rnn.BasicLSTMCell(1000)
        multi_cell = tf.contrib.rnn.MultiRNNCell([cell_1]  * 5 )
        cell_wrapped = tf.contrib.rnn.OutputProjectionWrapper(multi_cell, output_size=27)
        outputs, states = tf.nn.dynamic_rnn(cell_wrapped, inputs=to_be_fed, dtype=tf.float32)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=outputs[:,-1,:]))
        print("outputs:")
        print(np.shape(outputs))
        hold_prob = tf.placeholder(tf.float32, None)
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        steps = 5000

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(20000):
                batch_y, batch_x = self.get_lstm_batch(number_of_samples=50)

                sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.3})

                # PRINT OUT A MESSAGE EVERY 100 STEPS

                if i % 100 == 0:
                    print('Currently on step {}'.format(i))
                    print('Accuracy is:')
                    # Test the Train Model
                    matches = tf.equal(tf.argmax(outputs[:,-1,:]), tf.argmax(y_true))
                    acc = tf.reduce_mean(tf.cast(matches, tf.float32))
                    rnd = np.random.randint(0, 4)
                    print(sess.run(acc, feed_dict={x: self.smaller_images[rnd * 1000:(rnd + 1) * 1000],
                                                   y_true: self.smaller_oh[rnd * 1000:(rnd + 1) * 1000],
                                                   hold_prob: 1.0}))
                    print('\n')
            saver.save(sess, "lstm_conv_1chr/" + file_name)
        print("to be implemented")

    def gaussian_noise(self,image,intensity=123):
        image = np.reshape(image, newshape=(28, 28, 1))
        row, col, ch = image.shape
        mean = 0
        var = intensity
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return np.reshape(noisy,newshape=(28,28))

    def speckle_noise(self,image):
        image = np.reshape(image,newshape=(28,28,1))
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss * 0.2
        return np.reshape(noisy,newshape=(28,28))

    def make_smaller_img(self,image,small_nr=8):
        smaller_img = cv2.resize(np.reshape(image, newshape=(28, 28)), dsize=(28-small_nr, 28-small_nr))
        rand_start_top = np.random.randint(0,small_nr)
        rand_start_left = np.random.randint(0,small_nr)
        new_image = np.zeros((28,28))
        new_image[rand_start_left:28-(small_nr-rand_start_left),rand_start_top:28-(small_nr-rand_start_top)] = smaller_img
        return new_image

    def poison_noise(self,image):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    def sine_noise(self,image):
        new_image = []
        new_image = image
        A = np.shape(new_image)[0] / 3.0
        w = 2.0 / np.shape(new_image)[1]
        shift = lambda x: A * np.sin(0.1 * np.pi * x * w)

        for i in range(5,np.shape(new_image)[0]-5):
            new_image[:, i] = np.roll(new_image[:, i], int(shift(i)))
        return new_image

    def sp_noise(self,image,prob=0.1):
        output = np.zeros((28,28))
        prob = prob*10
        for i in range(np.shape(image)[0]):
            for j in range(np.shape(image)[1]):
                a = np.random.randint(0,10)
                if a < prob:
                    sp_rnd = np.random.randint(0,2)
                    if sp_rnd == 1:
                        output[i][j] = 255
                    else:
                        output[i][j] = 0
                else:
                    output[i][j] = image[i][j]
        return output

    def noise_data(self):
        def reinstantiate(image):
            img = []
            for i in range(28):
                line = []
                for j in range(28):
                    line.append(image[i][j])
                img.append(line)
            img = np.array(img)
            return img
        current_img = 0
        i=0
        initial_images = []
        for img in self.smaller_images:
            initial_images.append(img)
        initial_images = np.array(initial_images)
        for image in initial_images:
            small_image = reinstantiate(image)
            normal_image = reinstantiate(image)
            small_image = self.make_smaller_img(small_image)
            self.smaller_images.append(small_image)
            self.smaller_labels.append(self.smaller_labels[current_img])
            label_oh = np.zeros(27)
            label_oh[int(self.smaller_labels[current_img])-97] = 1
            self.smaller_oh.append(label_oh)
            i=i+1

            #for speckle
            # --> normal
            rnd_speckle = np.random.randint(0,2)
            if rnd_speckle == 0:
                pass
            elif rnd_speckle == 1:
                speckled_img = reinstantiate(normal_image)
                speckled_img = self.speckle_noise(speckled_img)
                self.smaller_images.append(speckled_img)
                self.smaller_labels.append(self.smaller_labels[current_img])
                label_oh = np.zeros(27)
                label_oh[int(self.smaller_labels[current_img]) - 97] = 1
                self.smaller_oh.append(label_oh)
                i=i+1
            # --> smaller
            rnd_speckle = np.random.randint(0, 2)
            if rnd_speckle == 0:
                pass
            elif rnd_speckle == 1:
                speckled_img =reinstantiate(small_image)
                speckled_img = self.speckle_noise(speckled_img)
                self.smaller_images.append(speckled_img)
                self.smaller_labels.append(self.smaller_labels[current_img])
                label_oh = np.zeros(27)
                label_oh[int(self.smaller_labels[current_img]) - 97] = 1
                self.smaller_oh.append(label_oh)
                i=i+1

            #for gauss
            # --> normal
            rnd_var = np.random.randint(50,140)
            rnd_true = np.random.randint(0,2)
            if rnd_true == 0:
                pass
            elif rnd_true == 1:
                gauss_img = reinstantiate(normal_image)
                gauss_img = self.gaussian_noise(gauss_img,intensity=rnd_var)
                self.smaller_images.append(gauss_img)
                self.smaller_labels.append(self.smaller_labels[current_img])
                label_oh = np.zeros(27)
                label_oh[int(self.smaller_labels[current_img]) - 97] = 1
                self.smaller_oh.append(label_oh)
                i=i+1
            # --> smaller
            rnd_true = np.random.randint(0,2)
            rnd_var = np.random.randint(50,150)
            if rnd_true == 0:
                pass
            elif rnd_true == 1:
                gauss_img = reinstantiate(small_image)
                gauss_img = self.gaussian_noise(gauss_img,intensity=rnd_var)
                self.smaller_images.append(gauss_img)
                self.smaller_labels.append(self.smaller_labels[current_img])
                label_oh = np.zeros(27)
                label_oh[int(self.smaller_labels[current_img]) - 97] = 1
                self.smaller_oh.append(label_oh)
                i=i+1

            #for poison
            # --> normal
            rnd_true = np.random.randint(0,2)
            if rnd_true == 0:
                pass
            elif rnd_true == 1:
                poisoned_img = reinstantiate(normal_image)
                poisoned_img = self.poison_noise(poisoned_img)
                self.smaller_images.append(poisoned_img)
                self.smaller_labels.append(self.smaller_labels[current_img])
                label_oh = np.zeros(27)
                label_oh[int(self.smaller_labels[current_img]) - 97] = 1
                self.smaller_oh.append(label_oh)
                i=i+1
            # --> smaller
            rnd_true = np.random.randint(0, 2)
            if rnd_true == 0:
                pass
            elif rnd_true == 1:
                gauss_img = reinstantiate(small_image)
                gauss_img = self.poison_noise(gauss_img)
                self.smaller_images.append(gauss_img)
                self.smaller_labels.append(self.smaller_labels[current_img])
                label_oh = np.zeros(27)
                label_oh[int(self.smaller_labels[current_img]) - 97] = 1
                self.smaller_oh.append(label_oh)
                i=i+1

            # for salt and pepper
            # --> normal
            prob = np.random.randint(0,10)
            prob = prob * 0.01
            rnd_true = np.random.randint(0, 2)
            if rnd_true == 0:
                pass
            elif rnd_true == 1:
                salt_and_peppered = reinstantiate(normal_image)
                salt_and_peppered = self.sp_noise(salt_and_peppered,prob=prob)
                self.smaller_images.append(salt_and_peppered)
                self.smaller_labels.append(self.smaller_labels[current_img])
                label_oh = np.zeros(27)
                label_oh[int(self.smaller_labels[current_img]) - 97] = 1
                self.smaller_oh.append(label_oh)
                i = i + 1
            # --> smaller
            prob = np.random.randint(0, 10)
            prob = prob * 0.01
            rnd_true = np.random.randint(0, 2)
            if rnd_true == 0:
                pass
            elif rnd_true == 1:
                salt_and_peppered = reinstantiate(small_image)
                salt_and_peppered = self.sp_noise(salt_and_peppered,prob=prob)
                self.smaller_images.append(salt_and_peppered)
                self.smaller_labels.append(self.smaller_labels[current_img])
                label_oh = np.zeros(27)
                label_oh[int(self.smaller_labels[current_img]) - 97] = 1
                self.smaller_oh.append(label_oh)
                i = i + 1

            # for sine
            rnd_true = np.random.randint(0,2)
            if rnd_true == 0:
                pass
            elif rnd_true == 1:
                sine_noised = reinstantiate(small_image)
                sine_noised = self.sine_noise(sine_noised)
                self.smaller_images.append(sine_noised)
                self.smaller_labels.append(self.smaller_labels[current_img])
                label_oh = np.zeros(27)
                label_oh[int(self.smaller_labels[current_img]) - 97] = 1
                self.smaller_oh.append(label_oh)
                i=i+1
            current_img = current_img + 1
        self.small_word_number = self.small_word_number + i
        print("Nr of data:")
        print(self.small_word_number)
a = CONV_LSTM()
a.get_in_order()
#a.create_1_chr_data()
a.append_data()
#a.noise_data()
a.shuffle_small_lsmt()
'''
for i in range(100):
    rnd_nr = np.random.randint(0,a.small_word_number)
    print(a.smaller_oh[rnd_nr])
    print(a.smaller_labels[rnd_nr])
    a.show_image_lettes(a.smaller_images[rnd_nr])
'''
a.model()
