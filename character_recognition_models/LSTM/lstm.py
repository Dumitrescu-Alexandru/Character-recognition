from sources.data import Data
import numpy as np
from cuvinte.word_generator import data_cleaning
import pickle
import os
import tensorflow as tf
from sources.utilities.convo_functions import init_weights
from sources.utilities.convo_functions import init_bias
from sources.utilities.convo_functions import conv2d
from sources.utilities.convo_functions import max_pool_2by2
from sources.utilities.convo_functions import convolutional_layer
from sources.utilities.convo_functions import normal_full_layer
PATH_TO_BINARIES = "C:\\Users\\alex_\\PycharmProjects\\Licenta\\LSTM\\binaries\\"
SLICE_DIM = 100
class LSTM(Data):

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

    def one_hot_encoding(self):

        oh_labels = []
        oh_current_label = []
        single_letter_label = []
        for string in self.lstm_data_labels:
            single_letter_label = np.zeros(27)
            for chr in string:
                if chr == " ":
                    single_letter_label[26] = 1
                else:
                    single_letter_label[ord(chr)-97] = 1
                oh_current_label.append(single_letter_label)
                single_letter_label = np.zeros(27)
            oh_labels.append(oh_current_label)

            oh_current_label = []
        self.lstm_data_labels_oh = oh_labels
        print(np.shape(self.lstm_data_labels_oh))

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

    def shuffle(self):
        a = self.letters_train
        b = self.letters_train_labels
        c = self.upper_lower_case

        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

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

    def shuffle_lstm(self):
        a = self.lstm_data_images
        b = self.lstm_data_labels
        c = self.lstm_data_labels_oh

        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)
        np.random.set_state(rng_state)
        np.random.shuffle(c)

    def create_words(self):
        string = ""
        max_nr = np.shape(self.words)[0]
        max_length = np.max([len(word) for word in self.words])
        random_word_nr = np.random.randint(max_nr)
        string = ''.join(self.words[random_word_nr])
        string+=" "
        while (len(string)<9):
            random_word_nr = np.random.randint(max_nr)
            str = ''.join(self.words[random_word_nr])
            if (len(str) + len(string)) < 16:
                string = string + str
        while(len(string)<16):
            string += " "
        return string

    def get_in_order(self):
        for i in range(self.train_letter_number):
            if self.upper_lower_case[i] == 0:
                self.order_lower[int(self.letter_labels[i])-1].append(self.letters_train[i])
            else:
                self.order_upper[int(self.letter_labels[i])-1].append(self.letters_train[i])

    def convert_to_images(self,string,first_upper = False):


        l = 0
        new_image = [[None] * len(string) * 28 * 28]
        new_image = np.float32(new_image)
        new_image = np.array(new_image)
        new_image = new_image.reshape(28, (28 * len(string)))
        z = ord(string[0])
        z = z - 97
        c = np.shape(self.order_upper[z])
        max = c[0] - 1
        from_nr = 0
        if first_upper:
            nr = np.random.randint(0, max - 1)
            new_image[0:28, l * 28:(l + 1) * 28] = self.order_upper[z][nr]
            l += 1
            from_nr = 1

        for i in string[from_nr:]:
            if i == " ":
                new_image[0:28, l * 28:(l + 1) * 28] = np.zeros((28,28))
                l += 1
            else:
                z = ord(i)
                z = z - 97
                c = np.shape(self.order_lower[z])
                max = c[0] - 1
                nr = np.random.randint(0, max - 1)
                new_image[0:28, l * 28:(l + 1) * 28] = self.order_lower[z][nr]
                l += 1
        #self.show_image_lettes(new_image)
        return new_image

    def add_images(self, images, file_name=PATH_TO_BINARIES+'word_images.bin'):

        created = os.path.isfile(file_name)
        if not created:
            self.create_file(file_name)
            file = open(file_name, 'wb')
        else:
            file = open(file_name, 'ab')
        pickle.dump(images, file)
        file.close()

    def add_labels(self, labels,lbls_file_name=PATH_TO_BINARIES+'word_labels.bin'):
        created_lbls = os.path.isfile(lbls_file_name)
        if not created_lbls:
            self.create_file(lbls_file_name)
            file_lbls = open(lbls_file_name, 'wb')
        else:
            file_lbls = open(lbls_file_name, 'ab')
        pickle.dump(labels, file_lbls)
        file_lbls.close()

    def create_file(self, name=PATH_TO_BINARIES+'word_images.bin'):
        file = open(name, 'wb')
        file.close()

    def read_images(self, name=PATH_TO_BINARIES+'word_images.bin'):
        fisier_imagini = open(name, 'rb')
        images = []
        while 1:
            try:
                images.append(pickle.load(fisier_imagini))
            except EOFError:
                break
        fisier_imagini.close()
        return images

    def read_labels(self, name=PATH_TO_BINARIES+'word_labels.bin'):
        fisier_labels = open(name, 'rb')
        labels = []
        while 1:
            try:
                labels.append(pickle.load(fisier_labels))
            except EOFError:
                break
        fisier_labels.close()
        return labels

    def generate_artificial_data(self,number_of_data=100,append = True):
        words_to_add = []
        images_to_add = []
        for i in range(number_of_data):
            words = self.create_words()
            image = self.convert_to_images(words)
            words_to_add.append(words)
            images_to_add.append(image)
        self.add_images(images_to_add)
        self.add_labels(words_to_add)

    def append_data(self,get_oh=True, show_some_data=False,show_data_with_oh=False):

        images = lstm_model.read_images()
        labels = lstm_model.read_labels()

        print(np.shape(images))
        print(np.shape(labels))

        for index in range(np.shape(images)[0]):
            for img in images[index]:
                self.lstm_data_images.append(img)
            for lbl in labels[index]:
                self.lstm_data_labels.append(lbl)

        self.words_training_number = np.shape(self.lstm_data_images)[0]
        if get_oh:
            self.one_hot_encoding()

        if show_some_data:

            for i in range(self.words_training_number):

                print(self.lstm_data_labels[i])
                if show_data_with_oh:
                    print(self.lstm_data_labels_oh[i])
                lstm_model.show_image_lettes(self.lstm_data_images[i])

    def get_lstm_batch(self, test_or_train='train', number_of_samples=100):
        lbls = []
        img_batches = []

        if number_of_samples + self.batch_placement >= self.small_word_number:
            lbls.append(self.lstm_data_labels_oh[self.batch_placement:self.small_word_number])
            img_batches.append(self.lstm_data_images[self.batch_placement:self.small_word_number])
            batches_to_go = self.small_word_number - self.batch_placement
            self.shuffle_lstm()
            lbls.append(self.lstm_data_labels_oh[0:batches_to_go])
            img_batches.append(self.lstm_data_images[0:batches_to_go])
            self.batch_placement = batches_to_go
        else:
            lbls.append(self.lstm_data_labels_oh[self.batch_placement:self.batch_placement + number_of_samples])
            img_batches.append(self.lstm_data_images[self.batch_placement:self.batch_placement + number_of_samples])
            self.batch_placement += number_of_samples
        return (lbls[0], img_batches[0])

    def create_smaller_data(self,size=5,nr_of_data=1000):

        words = [word for word in self.words if len(word) <= size]
        words = [wrd + " " * (size-len(wrd)) for wrd in words]
        unique_words = np.shape(words)[0]

        lbl_file_name = PATH_TO_BINARIES + "lbl_words_{0}_chrs.bin".format(size)
        img_file_name = PATH_TO_BINARIES + "img_words_{0}_chrs.bin".format(size)
        oh_labels = PATH_TO_BINARIES + "oh_lbl_{0}_chrs.bin".format(size)

        images = []
        labels = []
        labels_oh = []
        for i in range(nr_of_data):
            current_label = []
            rnd_nr = np.random.randint(0,unique_words)
            labels.append(words[rnd_nr])
            images.append(self.convert_to_images(words[rnd_nr]))
            for str in words[rnd_nr]:
                character_oh = np.zeros(27)
                if str == ' ':
                    character_oh[26] = 1
                else:
                    character_oh[ord(str) - 97] = 1
                current_label.append(character_oh)
            labels_oh.append(current_label)
        file = open(lbl_file_name, 'ab')
        pickle.dump(labels, file)
        file.close()
        file = open(img_file_name, 'ab')
        pickle.dump(images, file)
        file.close()
        file = open(oh_labels, 'ab')
        pickle.dump(labels_oh, file)
        file.close()

    def append_small_data(self,size=5,new_path = "C:\\Users\\alex_\\PycharmProjects\\Licenta\\cuvinte\\words_generate\\"):

        lbl_file_name = new_path + "lbl_words_{0}_chrs.bin".format(size)
        img_file_name = new_path + "img_words_{0}_chrs.bin".format(size)
        oh_labels = new_path + "oh_lbl_{0}_chrs.bin".format(size)

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

        label_word_file =open(lbl_file_name, 'rb')
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
        self.small_word_number = np.shape(imgs)[0]
        self.smaller_images = imgs
        self.smaller_labels = lbls
        self.smaller_oh = lbls_oh

    def reshape(self):
        self.lstm_data_images = np.array(self.lstm_data_images)
        print(np.shape(self.lstm_data_images))
        self.lstm_data_images = self.lstm_data_images.reshape(np.shape(self.lstm_data_images)[0],28*28*16)

    def create_the_model(self):
        num_intputs =  1
        x = tf.placeholder(tf.float32, shape=[None, 16*784])
        x_image = tf.reshape(x, [-1, 28, 28, 16])
        y_true = tf.placeholder(tf.float32, shape=[None, 16, 27])

        convo_1 = convolutional_layer(x_image, shape=[3, 3, 16, 32*16*16])
        convo_1_pooling = max_pool_2by2(convo_1)
        convo_2 = convolutional_layer(convo_1_pooling,shape = [3,3,32*16*16,27*16*4])
        convo_2_pooling = max_pool_2by2(convo_2)
        convo_3 = convolutional_layer(convo_2_pooling,shape= [3,3,27*16*4,16*4])
        convo_3_pooling = max_pool_2by2(convo_3)
        print(np.shape(convo_1))
        print(np.shape(convo_1_pooling))
        print(np.shape(convo_2))
        print(np.shape(convo_2_pooling))
        print(np.shape(convo_3))
        print(np.shape(convo_3_pooling))
        to_be_fed = tf.transpose(convo_3_pooling,perm=[0,3,1,2])
        to_be_fed = tf.reshape(to_be_fed,shape=[-1,16,64])
        print(np.shape(to_be_fed))
        # Just one feature, the time series
        num_inputs = 1
        num_time_steps = 16
        num_neurons = 64
        num_outputs = 16
        learning_rate = 0.03
        num_train_iterations = 4000
        batch_size = 1
        cell_1 = tf.contrib.rnn.BasicLSTMCell(64)
        initial_state = cell_1.zero_state(64, tf.float32)

        multi_cell = tf.contrib.rnn.MultiRNNCell([cell_1] * 16)
        cell_wrapped = tf.contrib.rnn.OutputProjectionWrapper(multi_cell, output_size=27)
        outputs, states = tf.nn.dynamic_rnn(cell_wrapped, inputs=to_be_fed, dtype=tf.float32)
        print(np.shape(outputs))
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=outputs))
        print(np.shape(y_true))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(20000):
                if i % 100 == 0:
                    print(i)
                    rand_nr = np.random.randint(0, 3000)
                    image = self.lstm_data_images[rand_nr]
                    print(self.lstm_data_labels[rand_nr])
                    a = np.reshape(image, newshape=[1, 784 * 16])
                    feed_dict = {x: a}
                    classification = sess.run(outputs, feed_dict)
                    #print(classification)
                    classification = classification[0]
                    string = ""
                    for j in range(16):
                        if (np.argmax(classification[j]) == 26):
                            string = string+ " "
                        else:
                            string = string + chr(97 + int(np.argmax(classification[j])))
                    print(string)
                batch_y, batch_x = self.get_lstm_batch(number_of_samples=10)
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y})
        '''
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(2000):
                batch_y, batch_x = self.get_lstm_batch(number_of_samples=50)

                sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})
                if i % 1000 == 0:
                    # Test the Train Model
                    rand_nr = np.random.randint(0,3000)
                    image = self.lstm_data_images[rand_nr]
                    print(self.lstm_data_labels[rand_nr])
                    a = np.reshape(image, newshape=[1, 784*16])
                    feed_dict = {x: a}
                    classification = sess.run(y_pred, feed_dict)
                    classification = classification[0]
                    for j in range(16):
                        if(np.argmax(classification[j*27:(j+1)*27]) == 26):
                            print(" ")
                        else:
                            print(chr(97 + int(np.argmax(classification[j*27:(j+1)*27] - 27*j))))
        '''

    def get_small_batch(self,number_of_samples=100):
        lbls = []
        img_batches = []

        if number_of_samples + self.small_batch_placement >= self.small_word_number:
            lbls.append(self.smaller_oh[self.small_batch_placement:self.small_word_number])
            img_batches.append(self.smaller_images[self.small_batch_placement:self.small_word_number])
            batches_to_go = self.small_word_number - self.small_batch_placement
            self.shuffle_small_lsmt()
            lbls.append(self.lstm_data_labels_oh[0:batches_to_go])
            img_batches.append(self.smaller_images[0:batches_to_go])
            self.small_batch_placement = batches_to_go
        else:
            lbls.append(self.smaller_oh[self.small_batch_placement:self.small_batch_placement + number_of_samples])
            img_batches.append(self.smaller_images[self.small_batch_placement:self.small_batch_placement + number_of_samples])
            self.small_batch_placement += number_of_samples
        return (lbls[0], img_batches[0])

    def small_model(self,word_nr=5,file_name="{0}_chrs_conv_lstm_model.ckpt".format(5)):

        x = tf.placeholder(tf.float32, shape=[None, 28, word_nr * 28])

        x_image = tf.reshape(x, [-1, 28, 28, 5])

        y_true = tf.placeholder(tf.float32, shape=[None,word_nr, 27])

        convo_1 = convolutional_layer(x_image, shape=[3, 3, 5, 32 * word_nr * word_nr])
        convo_1_pooling = max_pool_2by2(convo_1)
        convo_2 = convolutional_layer(convo_1_pooling, shape=[3, 3, 32 * word_nr * word_nr, 27 * word_nr * word_nr])
        convo_2_pooling = max_pool_2by2(convo_2)
        convo_3 = convolutional_layer(convo_2_pooling, shape=[3, 3, 27 * word_nr * word_nr, word_nr * 4])
        convo_3_pooling = max_pool_2by2(convo_3)
        to_be_fed = tf.transpose(convo_3_pooling, perm=[0, 3, 1, 2])
        to_be_fed = tf.reshape(to_be_fed, shape=[-1, word_nr, 64])
        cell_1 = tf.contrib.rnn.BasicLSTMCell(64)
        multi_cell = tf.contrib.rnn.MultiRNNCell([cell_1] * word_nr)
        cell_wrapped = tf.contrib.rnn.OutputProjectionWrapper(multi_cell, output_size=27)
        outputs, states = tf.nn.dynamic_rnn(cell_wrapped, inputs=to_be_fed, dtype=tf.float32)
        print(np.shape(outputs))
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=outputs))
        print(np.shape(y_true))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(cross_entropy)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        saver = tf.train.Saver()
        file = open("test_results/rezultate.txt", "w")
        file.close()
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(200000):
                from matplotlib import pyplot
                import matplotlib as mpl
                pyplot.close("all")
                fig = pyplot.figure()
                ax = fig.add_subplot(1, 1, 1)

                if i % 1000 == 0 and i != 0 :
                    file = open("test_results/rezultate.txt", "a")
                    print(i)
                    rand_nr = np.random.randint(0, 30000)
                    image = self.smaller_images[rand_nr]
                    print(self.smaller_labels[rand_nr])
                    a = np.reshape(image, newshape=[1, 28 ,28* word_nr])
                    print(np.shape(image))
                    imgplot = ax.imshow(image, cmap=mpl.cm.gray)
                    imgplot.set_interpolation('nearest')
                    ax.xaxis.set_ticks_position('top')
                    ax.yaxis.set_ticks_position('left')
                    pyplot.savefig("test_results/image_nr_{0}.jpg".format(i))
                    pyplot.close('all')
                    feed_dict = {x: a}
                    classification = sess.run(outputs, feed_dict)
                    # print(classification)
                    classification = classification[0]
                    string = ""
                    for j in range(5):
                        if (np.argmax(classification[j]) == 26):
                            string = string + " "
                        else:
                            string = string + chr(97 + int(np.argmax(classification[j])))
                    file.write("for image number "+str(i)+": "+string)
                    file.write("\ntrue string: " + self.smaller_labels[rand_nr])
                    file.write("\n")
                    file.close()
                batch_y, batch_x = self.get_small_batch(number_of_samples=30)
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y})
            saver.save(sess,"./"+file_name)

    def load_model(self,word_nr=5,file_name = "{0}_chrs_conv_lstm_model.ckpt".format(5)):
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=[None, 28, word_nr * 28])
        x_image = tf.reshape(x, [-1, 28, 28, 5])
        y_true = tf.placeholder(tf.float32, shape=[None,word_nr, 27])
        convo_1 = convolutional_layer(x_image, shape=[3, 3, 5, 32 * word_nr * word_nr])
        convo_1_pooling = max_pool_2by2(convo_1)
        convo_2 = convolutional_layer(convo_1_pooling, shape=[3, 3, 32 * word_nr * word_nr, 27 * word_nr * word_nr])
        convo_2_pooling = max_pool_2by2(convo_2)
        convo_3 = convolutional_layer(convo_2_pooling, shape=[3, 3, 27 * word_nr * word_nr, word_nr * 4])
        convo_3_pooling = max_pool_2by2(convo_3)
        cell_1 = tf.contrib.rnn.BasicLSTMCell(64)
        to_be_fed = tf.transpose(convo_3_pooling, perm=[0, 3, 1, 2])
        to_be_fed = tf.reshape(to_be_fed, shape=[-1, word_nr, 64])
        multi_cell = tf.contrib.rnn.MultiRNNCell([cell_1] * word_nr)
        cell_wrapped = tf.contrib.rnn.OutputProjectionWrapper(multi_cell, output_size=27)
        outputs, states = tf.nn.dynamic_rnn(cell_wrapped, inputs=to_be_fed, dtype=tf.float32)
        print(np.shape(outputs))
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=outputs))
        print(np.shape(y_true))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, r"./" + file_name)
            image = self.smaller_images[0]
            image = np.reshape(image, newshape=[28, 28*5])
            self.show_image_lettes(image)

            a = np.reshape(image, newshape=[1, 28,28*5])
            feed_dict = {x: a}
            classification = sess.run(outputs, feed_dict)
            classification = classification[0]
            string = ""
            for j in range(5):
                if (np.argmax(classification[j]) == 26):
                    string = string + " "
                else:
                    string = string + chr(97 + int(np.argmax(classification[j])))
            print(string)

    def small_model_var(self,word_nr=3,file_name="{0}_chrs_conv_lstm_model.ckpt".format(3)):
        num_intputs = 1

        x = tf.placeholder(tf.float32, shape=[None, 28, word_nr * 28])

        x_image = tf.reshape(x, [-1, 28, 28*3, 1])

        y_true = tf.placeholder(tf.float32, shape=[None, word_nr, 27])
        convo_1 = convolutional_layer(x_image, shape=[5, 5, 1, 32])
        convo_1_pooling = max_pool_2by2(convo_1)
        convo_2 = convolutional_layer(convo_1_pooling, shape=[3, 3, 32, 64])
        convo_2_pooling = max_pool_2by2(convo_2)
        convo_3 = convolutional_layer(convo_2_pooling, shape=[3, 3, 64, 27])
        convo_3_pooling = max_pool_2by2(convo_3)
        to_be_fed = tf.reshape(convo_3_pooling, shape=[-1, 4*11*3, 9])
        cell_1 = tf.contrib.rnn.BasicLSTMCell(3*44)
        cell_2 = tf.contrib.rnn.BasicLSTMCell(3*44)
        cell_3 = tf.contrib.rnn.BasicLSTMCell(3*44)
        cell_4 = tf.contrib.rnn.BasicLSTMCell(3*44)
        cell_5 = tf.contrib.rnn.BasicLSTMCell(3*44)
        cell_6 = tf.contrib.rnn.BasicLSTMCell(3*44)
        cell_7 = tf.contrib.rnn.BasicLSTMCell(3*44)
        cell_8 = tf.contrib.rnn.BasicLSTMCell(3*44)
        cell_9 = tf.contrib.rnn.BasicLSTMCell(3*44)
        multi_cell = tf.contrib.rnn.MultiRNNCell([cell_1,cell_2,cell_3,cell_4,cell_5,cell_6,cell_7,cell_8,cell_9])
        cell_wrapped = tf.contrib.rnn.OutputProjectionWrapper(multi_cell, output_size=27)
        outputs, states = tf.nn.dynamic_rnn(cell_wrapped, inputs=to_be_fed, dtype=tf.float32)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true, logits=tf.transpose([outputs[:,2,:],outputs[:,5,:],outputs[:,8,:]],(1,0,2))))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(cross_entropy)
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        saver = tf.train.Saver()
        file = open("test_results_3chrs/rezultate.txt", "w")
        file.close()
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(20000):
                from matplotlib import pyplot
                import matplotlib as mpl
                pyplot.close("all")
                fig = pyplot.figure()
                ax = fig.add_subplot(1, 1, 1)

                if i % 100 == 0:
                    file = open("test_results_3chrs/rezultate.txt", "a")
                    print(i)
                    rand_nr = np.random.randint(0, 20000)
                    image = self.smaller_images[rand_nr]
                    a = np.reshape(image, newshape=[1, 28, 28 * word_nr])
                    imgplot = ax.imshow(image, cmap=mpl.cm.gray)
                    imgplot.set_interpolation('nearest')
                    ax.xaxis.set_ticks_position('top')
                    ax.yaxis.set_ticks_position('left')
                    pyplot.savefig("test_results_3chrs/image_nr_{0}.jpg".format(i))
                    pyplot.close('all')
                    feed_dict = {x: a}
                    classification = sess.run(outputs, feed_dict)

                    interested_outputs = [classification[:,2,:],classification[:,5,:],classification[:,8,:]]
                    interested_outputs = np.transpose(interested_outputs,(1,0,2))
                    interested_outputs = interested_outputs[0]
                    print(np.shape(classification))
                    print(np.shape(interested_outputs))

                    string = ""
                    for j in range(3):
                        if (np.argmax(interested_outputs[j]) == 26):
                            string = string + " "
                        else:
                            string = string + chr(97 + int(np.argmax(interested_outputs[j])))
                    file.write("for image number " + str(i) + ": " + string)
                    file.write("\ntrue string: " + self.smaller_labels[rand_nr])
                    file.write("\n")
                    file.close()
                batch_y, batch_x = self.get_small_batch(number_of_samples=30)
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y})
            saver.save(sess, "3chr_model/" + file_name)

    def small_model_20(self,word_nr=3,file_name="3chr_model_20//{0}_chrs_conv_lstm_model.ckpt".format(3)):
        num_intputs = 1
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=[None, 28, word_nr * 28])

        x_image = tf.reshape(x, [-1, 28, 28, 3])

        y_true = tf.placeholder(tf.float32, shape=[None, word_nr * 27])

        convo_1 = convolutional_layer(x_image, shape=[5, 5, word_nr, 30 * word_nr * word_nr])
        convo_1_pooling = max_pool_2by2(convo_1)
        convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 30 * word_nr * word_nr, 15 * word_nr * word_nr])
        convo_2_pooling = max_pool_2by2(convo_2)
        convo_3 = convolutional_layer(convo_2_pooling, shape=[5, 5, 15 * word_nr * word_nr, 135])
        convo_3_pooling = max_pool_2by2(convo_3)

        print(np.shape(convo_1))
        print(np.shape(convo_1_pooling))
        print(np.shape(convo_2))
        print(np.shape(convo_2_pooling))
        print(np.shape(convo_3))
        print("convo_3_pooling:")
        print(np.shape(convo_3_pooling))

        to_be_fed = tf.transpose(convo_3_pooling, perm=[0, 3, 1, 2])
        to_be_fed = tf.reshape(to_be_fed, shape=[-1, word_nr , 720])
        print(np.shape(to_be_fed))
        # Just one feature, the time series
        num_inputs = 1
        num_time_steps = 16
        num_neurons = 64
        num_outputs = 16
        learning_rate = 0.03
        num_train_iterations = 4000
        batch_size = 1
        cell_1 = tf.contrib.rnn.BasicLSTMCell(word_nr*720)
        initial_state = cell_1.zero_state(word_nr*720, tf.float32)
        cell_2 = tf.contrib.rnn.BasicLSTMCell(word_nr*180)
        cell_3 = tf.contrib.rnn.BasicLSTMCell(word_nr*60)
        cells = [cell_1,cell_2,cell_3]
        multi_cell = tf.contrib.rnn.MultiRNNCell(cells)
        cell_wrapped = tf.contrib.rnn.OutputProjectionWrapper(multi_cell, output_size=27*3)
        outputs, states = tf.nn.dynamic_rnn(cell_wrapped, inputs=to_be_fed, dtype=tf.float32)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=outputs[:, -1, :]))
        print("y_true and outputs:")
        print(np.shape(outputs))
        print(np.shape(y_true))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        saver = tf.train.Saver()
        file = open("test_results_3chrs_20/rezultate.txt", "w")
        file.close()
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(600000):
                from matplotlib import pyplot
                import matplotlib as mpl
                pyplot.close("all")
                fig = pyplot.figure()
                ax = fig.add_subplot(1, 1, 1)

                if i % 1000 == 0 and i != 0:
                    file = open("test_results_3chrs_20/rezultate.txt", "a")
                    print(i)
                    rand_nr = np.random.randint(0, 20000)
                    image = self.smaller_images[rand_nr]
                    a = np.reshape(image, newshape=[1, 28, 28 * word_nr])
                    imgplot = ax.imshow(image, cmap=mpl.cm.gray)
                    imgplot.set_interpolation('nearest')
                    ax.xaxis.set_ticks_position('top')
                    ax.yaxis.set_ticks_position('left')
                    pyplot.savefig("test_results_3chrs_20/image_nr_{0}.jpg".format(i))
                    pyplot.close('all')
                    feed_dict = {x: a}
                    classification = sess.run(outputs, feed_dict)
                    # print(classification)
                    classification = classification[:, -1, :]
                    classification = np.reshape(classification,newshape=(3,27))
                    string = ""
                    for j in range(3):
                        if (np.argmax(classification[j]) == 26):
                            string = string + " "
                        else:
                            string = string + chr(97 + int(np.argmax(classification[j])))
                    file.write("for image number " + str(i) + ": " + string)
                    file.write("\ntrue string: " + self.smaller_labels[rand_nr])
                    file.write("\n")
                    file.close()
                batch_y, batch_x = self.get_small_batch(number_of_samples=40)
                sess.run(train, feed_dict={x: batch_x, y_true: np.reshape(batch_y,newshape=(-1,81))})
            saver.save(sess,  file_name)


lstm_model = LSTM()
'''
for i in range(15):
    x = np.random.randint(9000)
    print(lstm_model.letters_train_labels[x])
    print(lstm_model.upper_lower_case[x])
    lstm_model.show_image_lettes(lstm_model.letters_train[x])



for i in range(0,26):
    print(np.shape(lstm_model.order_lower[i]))

    lstm_model.show_image_lettes(lstm_model.order_lower[i][0])
    index = np.shape(lstm_model.order_lower[i])[0]
    index = index-1
    lstm_model.show_image_lettes(lstm_model.order_lower[i][index])

    print(np.shape(lstm_model.order_upper[i]))

    lstm_model.show_image_lettes(lstm_model.order_upper[i][0])
    index = np.shape(lstm_model.order_upper[i])[0]
    index = index - 1
    lstm_model.show_image_lettes(lstm_model.order_upper[i][index])

for i in lstm_model.words:
    print(i)

print (lstm_model.words)
'''


#lstm_model.get_in_order()


lstm_model.append_data(show_some_data=False,show_data_with_oh = True)
lstm_model.one_hot_encoding()

lstm_model.append_small_data(size=3)


lstm_model.small_model_var()
