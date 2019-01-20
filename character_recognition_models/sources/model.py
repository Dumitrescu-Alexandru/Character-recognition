import os
import numpy as np
import tensorflow as tf
from sources.data import Data
from sources.utilities.convo_functions import init_weights
from sources.utilities.convo_functions import init_bias
from sources.utilities.convo_functions import conv2d
from sources.utilities.convo_functions import max_pool_2by2
from sources.utilities.convo_functions import convolutional_layer
from sources.utilities.convo_functions import normal_full_layer

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

class Model(Data):
    combined_images_train = []
    combined_labels_train = []
    combined_images_test = []
    combined_labels_test = []

    type = None
    digits_or_letters = 'digit'

    def __init__(self, type='simple', combined_together=False,shuffle=False):
        super().__init__()
        self.type = type

        # CAREFUL WITH SHUFFLING NOW... USE COMBINE SHUFFLE?
        # THINK WE NEED A GET BATCH COMBINED AS WELL...
        self.convert_to_onehot('d', 'train')
        self.convert_to_onehot('d', 'test')
        self.convert_to_onehot('l', 'train')
        self.convert_to_onehot('l', 'test')
        # need a combined together argument
        if combined_together:
            self.combined_labels_train, self.combined_labels_test, self.combined_images_train, self.combined_images_test = self.combine_letters_and_digits()
            if shuffle:
                self.shuffle_together()
            self.combined_images_train = self.combined_images_train / 255
            self.combined_images_test = self.combined_images_test / 255

    def shuffle_together(self):
        a = self.combined_images_train
        b = self.combined_labels_train
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    def vectorize(self):
        self.digit_images = self.digit_images.reshape(60000, 784)
        self.digit_images_test = self.digit_images_test.reshape(10000, 784)
        self.letter_images = self.letter_images.reshape(124800, 784)
        self.letter_images_test = self.letter_images_test.reshape(20800, 784)

    def combined_reshape(self):
        if np.shape(self.combined_images_train) == (184800, 28, 28):
            self.combined_images_train = self.combined_images_train.reshape(184800, 784)
            self.combined_images_test = self.combined_images_test.reshape(30800, 784)
        elif np.shape(self.combined_images_train) == (184800, 784):
            self.combined_images_train = self.combined_images_train.reshape(184800, 28, 28)
            self.combined_images_test = self.combined_images_test.reshape(30800, 28, 28)

    def simple_train(self, type='digit'):
        if type == 'digit':
            x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
            w = tf.Variable(tf.zeros([784, 10]), name='w')
            b = tf.Variable(tf.zeros([10]), name='b')
            y = tf.matmul(x, w) + b
            y_true = tf.placeholder(tf.float32, [None, 10], name='y_true')
        else:
            x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
            w = tf.Variable(tf.zeros([784, 26]), name='w')
            b = tf.Variable(tf.zeros([26]), name='b')
            y = tf.matmul(x, w) + b
            y_true = tf.placeholder(tf.float32, [None, 26], name='y_true')
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for step in range(10000):
                if type == 'digit':
                    batch_y, batch_x = self.get_batch(type='d', number_of_samples=10)
                else:
                    batch_y, batch_x = self.get_batch(type='l', number_of_samples=10)
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

            # evaluate
            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))

            # [true, false, true ...] -> [1, 0, 1...]
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            if type == 'digit':
                result = sess.run(acc, feed_dict={x: self.digit_images, y_true: self.digit_labels})
            else:
                result = sess.run(acc, feed_dict={x: self.letter_images, y_true: self.letter_labels})
        print(result)

    def show_image_combined(self, img_nr, test_or_train='train'):
        if test_or_train == 'train':
            image = self.combined_images_train[img_nr]
        elif test_or_train == 'test':
            image = self.combined_images_test[img_nr]
        from matplotlib import pyplot
        import matplotlib as mpl
        fig = pyplot.figure()
        ax = fig.add_subplot(1, 1, 1)
        imgplot = ax.imshow(image, cmap=mpl.cm.gray)
        imgplot.set_interpolation('nearest')
        ax.xaxis.set_ticks_position('top')
        ax.yaxis.set_ticks_position('left')
        pyplot.show()

    def get_batch_combined(self, number_of_samples=100):
        lbls = []
        img_batches = []
        if number_of_samples + self.batch_placements[2] >= 184800:
            lbls.append(self.combined_labels_train[self.batch_placements[2]:184800])
            img_batches.append(self.combined_images_train[self.batch_placements[2]:184800])
            batches_to_go = 184800 - self.batch_placements[2]
            self.shuffle_together()
            lbls.append(self.combined_labels_train[0:batches_to_go])
            img_batches.append(self.combined_images_train[0:batches_to_go])
            self.batch_placements[2] = batches_to_go
        else:
            lbls.append(
                self.combined_labels_train[self.batch_placements[2]:self.batch_placements[2] + number_of_samples])
            img_batches.append(
                self.combined_images_train[self.batch_placements[2]:self.batch_placements[2] + number_of_samples])
            self.batch_placements[2] += number_of_samples
        return (lbls[0], img_batches[0])

    def simple_architecture(self, steps=1000, samples_number=100):

        x = tf.placeholder(tf.float32, shape=[None, 784], name='x')
        w = tf.Variable(tf.zeros([784, 36]), name='w')
        b = tf.Variable(tf.zeros([36]), name='b')
        y = tf.matmul(x, w) + b
        y_true = tf.placeholder(tf.float32, [None, 36], name='y_true')
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for step in range(steps):
                batch_y, batch_x = self.get_batch_combined(number_of_samples=samples_number)
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

            correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
            acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
            result = sess.run(acc, feed_dict={x: self.combined_images_train, y_true: self.combined_labels_train})

            on_test_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
            acc_on_test = tf.reduce_mean(tf.cast(on_test_pred,tf.float32))
            result_on_test = sess.run(acc_on_test,feed_dict={x: self.combined_images_test,y_true:self.combined_labels_test})

            #print(self.evaluate_on_test(w,b))

        print("Prediction result on train data:")
        print(result)
        print("Prediction result on test data:")
        print(result_on_test)

    def convolutional_architecture(self,file_name='conv_model.ckpt'):



        x = tf.placeholder(tf.float32,shape=[None,784])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        y_true = tf.placeholder(tf.float32,shape=[None,36])

        convo_1 = convolutional_layer(x_image,shape=[3,3,1,32])
        convo_1_pooling = max_pool_2by2(convo_1)
        convo_2 = convolutional_layer(convo_1_pooling,shape=[5,5,32,64])
        convo_2_pooling = max_pool_2by2(convo_2)
        convo_2_flat = tf.reshape(convo_2_pooling, shape=[-1, 7*7*64 ])
        full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

        hold_prob = tf.placeholder(tf.float32, None)
        full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)
        y_pred = normal_full_layer(full_layer_one, 36)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        steps = 5000
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)

        saver = tf.train.Saver()

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            for i in range(20000):
                batch_y, batch_x = self.get_batch_combined(number_of_samples=50)

                sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 0.5})

                # PRINT OUT A MESSAGE EVERY 100 STEPS

                if i % 100 == 0:
                    print('Currently on step {}'.format(i))
                    print('Accuracy is:')
                    # Test the Train Model
                    matches = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))

                    acc = tf.reduce_mean(tf.cast(matches, tf.float32))
                    print(np.shape(self.combined_labels_test))
                    print(np.shape(self.combined_images_test))
                    rnd = np.random.randint(0,25)
                    print(sess.run(acc, feed_dict={x: self.combined_images_test[rnd*1000:(rnd+1)*1000], y_true: self.combined_labels_test[rnd*1000:(rnd+1)*1000], hold_prob: 1.0}))
                    print('\n')
            saver.save(sess,"models/"+file_name)
        print("to be implemented")

    def load_model(self,type='conv_model',file_name='conv_model.ckpt',demo=False,demo_nr=100):
        if type == 'conv_model':
            x = tf.placeholder(tf.float32, shape=[None, 784])
            x_image = tf.reshape(x, [-1, 28, 28, 1])
            y_true = tf.placeholder(tf.float32, shape=[None, 36])

            convo_1 = convolutional_layer(x_image, shape=[5, 5, 1, 32])
            convo_1_pooling = max_pool_2by2(convo_1)
            convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 32, 64])
            convo_2_pooling = max_pool_2by2(convo_2)
            convo_2_flat = tf.reshape(convo_2_pooling, shape=[-1, 7 * 7 * 64])
            full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

            hold_prob = tf.placeholder(tf.float32, None)
            full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=hold_prob)
            y_pred = normal_full_layer(full_layer_one, 36)
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
            optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
            train = optimizer.minimize(cross_entropy)
            init = tf.global_variables_initializer()

            saver = tf.train.Saver()

            some_data = Data()

            with tf.Session() as sess:
                sess.run(init)
                saver.restore(sess, "models/" + file_name)
                if demo:
                    for i in range(demo_nr):
                        image = np.reshape(some_data.letter_images[i], newshape=[28, 28])
                        from matplotlib import pyplot
                        import matplotlib as mpl
                        fig = pyplot.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        imgplot = ax.imshow(image, cmap=mpl.cm.gray)
                        imgplot.set_interpolation('nearest')
                        ax.xaxis.set_ticks_position('top')
                        ax.yaxis.set_ticks_position('left')
                        pyplot.show()
                        a = np.reshape(some_data.letter_images[i], newshape=[1, 784])
                        feed_dict = {x: a}
                        classification = sess.run(y_pred, feed_dict)
                        print(np.argmax(classification))
                    for i in range(demo_nr):
                        image = np.reshape(some_data.digit_images[i], newshape=[28, 28])
                        from matplotlib import pyplot
                        import matplotlib as mpl
                        fig = pyplot.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        imgplot = ax.imshow(image, cmap=mpl.cm.gray)
                        imgplot.set_interpolation('nearest')
                        ax.xaxis.set_ticks_position('top')
                        ax.yaxis.set_ticks_position('left')
                        pyplot.show()
                        a = np.reshape(some_data.digit_images[i], newshape=[1, 784])
                        feed_dict = {x: a}
                        classification = sess.run(y_pred, feed_dict)
                        print(np.argmax(classification))