import os
import numpy as np
import tensorflow as tf
from sources.data import Data
import cv2
from sources.utilities.convo_functions import init_weights
from sources.utilities.convo_functions import init_bias
from sources.utilities.convo_functions import conv2d
from sources.utilities.convo_functions import max_pool_2by2
from sources.utilities.convo_functions import convolutional_layer
from sources.utilities.convo_functions import normal_full_layer

# change for the EMNIST folder path in your computer
PATH_TO_BINARIES = "C:\\Users\\alex_\\Desktop\\Licenta Dumitrescu Alexandru\\Proiect Licenta\\LSTM\\binaries\\"
PATH = "C:\\Users\\alex_\\Desktop\\Licenta Dumitrescu Alexandru\\Proiect Licenta\\EMNIST"
letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']

TRAIN_IMG_DIGITS = os.path.join(PATH,'train-images-idx3-ubyte')
TRAIN_LBL_DIGITS = os.path.join(PATH,'train-labels-idx1-ubyte')
TEST_IMG_DIGITS  = os.path.join(PATH,'t10k-images-idx3-ubyte')
TEST_LBL_DIGITS  = os.path.join(PATH,'t10k-labels-idx1-ubyte')

TRAIN_IMG_LETTERS = os.path.join(PATH,'emnist-letters-train-images-idx3-ubyte')
TRAIN_LBL_LETTERS = os.path.join(PATH,'emnist-letters-train-labels-idx1-ubyte')
TEST_IMG_LETTERS  = os.path.join(PATH,'emnist-letters-test-images-idx3-ubyte')
TEST_LBL_LETTERS  = os.path.join(PATH,'emnist-letters-test-labels-idx1-ubyte')



class Model_Documentatie(Data):
    combined_images_train = []
    combined_labels_train = []
    combined_images_test = []
    combined_labels_test = []
    smaller_images = []
    smaller_oh = []
    smaller_labels = []
    small_word_number = 0
    type = None
    digits_or_letters = 'digit'

    def __init__(self, type='simple', combined_together=False,shuffle=False):
        super().__init__()
        self.type = type
        self.digit_images = self.digit_images/255
        self.digit_images_test = self.digit_images_test/255

        # CAREFUL WITH SHUFFLING NOW... USE COMBINE SHUFFLE?
        # THINK WE NEED A GET BATCH COMBINED AS WELL...
        self.convert_to_onehot('d', 'train')
        self.convert_to_onehot('d', 'test')
        self.convert_to_onehot('l', 'train')
        self.convert_to_onehot('l', 'test')

    # simpla metoda penntru a afisa un plot matriceal al imaginii
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

    # simpla metoda pentru a "amesteca" imaginile.
    def shuffle_together(self):
        a = self.digit_images
        b = self.digit_labels
        rng_state = np.random.get_state()
        np.random.shuffle(a)
        np.random.set_state(rng_state)
        np.random.shuffle(b)

    # arhitectura fully connected
    def simple_architecture(self, steps=20000, samples_number=100):

        file_test = open("test_results.txt",'a')
        file_train = open("train_results.txt",'a')
        x = tf.placeholder(tf.float32, shape=[None, 28, 28], name='x')
        w = tf.Variable(tf.zeros([784, 10]), name='w')
        # vectorizarea imaginii
        x_image = tf.reshape(x,shape=(-1,784))
        b = tf.Variable(tf.zeros([10]), name='b')

        # definirea modelului (W*X + b)
        y = tf.matmul(x_image, w) + b
        y_true = tf.placeholder(tf.float32, [None, 10], name='y_true')
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y))
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            sess.run(init)

            for step in range(steps):
                batch_y, batch_x = self.get_batch(number_of_samples=samples_number)
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y})

                # o data la 100 de iteratii, salveaza si afiseaza rezultatele
                # (acuratetea pe setul de train/test)
                if step % 100 == 0 and step != 0:
                    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_true, 1))
                    acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                    result = sess.run(acc, feed_dict={x: self.digit_images, y_true: self.digit_labels})

                    on_test_pred = tf.equal(tf.argmax(y,1),tf.argmax(y_true,1))
                    acc_on_test = tf.reduce_mean(tf.cast(on_test_pred,tf.float32))
                    result_on_test = sess.run(acc_on_test,feed_dict={x: self.digit_images_test,y_true:self.digit_labels_test})
                    print("On step {0}".format(step))
                    print("Results on train:{0}".format(result))
                    print("Results on test:{0}".format(result_on_test))
                    file_test.write(str(result_on_test) + " ")
                    file_train.write(str(result) + " ")

        file_test.close()
        file_train.close()

    # arhitectura convolutiva folosita si in aplicatie
    def conv_architecture(self, steps=20000, samples_number=100,file_name='C:\\Users\\alex_\\PycharmProjects\\Licenta\\sources\\1chr_digits_model_conv.ckpt'):

        file_test = open("test_results.txt", 'a')
        file_train = open("train_results.txt", 'a')
        x = tf.placeholder(tf.float32, shape=[None, 28, 28], name='x')
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        y_true = tf.placeholder(tf.float32, [None, 10], name='y_true')

        # definirea arhitecturii
        convo_1 = convolutional_layer(x_image, shape=[6, 6, 1, 32])
        convo_1_pooling = max_pool_2by2(convo_1)
        convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 32, 64])
        convo_2_pooling = max_pool_2by2(convo_2)
        convo_2_flat = tf.reshape(convo_2_pooling, shape=[-1, 7 * 7 * 64])
        full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

        # definirea dropout <- valoarea lui este data in session
        hold_prob = tf.placeholder(tf.float32)
        full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=0.5)
        y_pred = normal_full_layer(full_one_dropout, 10)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(cross_entropy)

        # doar pentru utilizarea GPU-ului. Stergeti argumentul de la linia 150 pentru procesor
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
            sess.run(init)

            for step in range(steps):
                batch_y, batch_x = self.get_batch(number_of_samples=samples_number)
                sess.run(train, feed_dict={x:batch_x,y_true:batch_y,hold_prob:0.7})

                # o data la 100 de pasi, se afiseaza si salveaza acuratetea
                if step % 100 == 0 and step != 0:
                    res_test = 0
                    res_train = 0
                    for i in range(10):
                        correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
                        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
                        result = sess.run(acc, feed_dict={x: self.digit_images[i*1000:(i+1)*1000], y_true: self.digit_labels[i*1000:(i+1)*1000]})
                        res_train = res_train + result
                    for i in range(10):
                        on_test_pred = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y_true, 1))
                        acc_on_test = tf.reduce_mean(tf.cast(on_test_pred, tf.float32))
                        result_on_test = sess.run(acc_on_test,
                                                      feed_dict={x: self.digit_images_test[i*1000:(i+1)*1000], y_true: self.digit_labels_test[i*1000:(i+1)*1000]})
                        res_test = res_test + result_on_test
                    res_test = res_test/10
                    res_train = res_train/10
                    print("On step {0}".format(step))
                    print("Results on train:{0}".format(result))
                    print("Results on test:{0}".format(result_on_test))
                    file_test.write(str(result_on_test) + " ")
                    file_train.write(str(result) + " ")
            saver.save(sess, file_name)

        file_test.close()
        file_train.close()

    def gaussian_noise(self,image,variance=123):
        image = np.reshape(image, newshape=(28, 28, 1))
        row, col, ch = image.shape
        mean = 0
        var = variance
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
        smaller_img = cv2.resize(np.reshape(image, newshape=(28, 28)),
                                 dsize=(28-small_nr, 28-small_nr))
        rand_start_top = np.random.randint(0,small_nr)
        rand_start_left = np.random.randint(0,small_nr)
        new_image = np.zeros((28,28))
        new_image[rand_start_left:28-(small_nr-rand_start_left),
                    rand_start_top:28-(small_nr-rand_start_top)] = smaller_img
        return new_image

    def poison_noise(self,image):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy

    def sine_noise(self,image):
        new_image = image
        A = np.shape(new_image)[0] / 3.0
        w = 2.0 / np.shape(new_image)[1]
        shift = lambda x: A * np.sin(0.05 * np.pi * x * w)

        for i in range(5,np.shape(new_image)[0]):
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



# cand este rulat, programul antreneaza o noura retea convoluitva
model = Model_Documentatie()
model.shuffle_together()
model.conv_architecture()


