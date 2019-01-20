import numpy as np
import tensorflow as tf
from convo_functions import convolutional_layer
from convo_functions import max_pool_2by2
from convo_functions import normal_full_layer
class Predictions():

    def convo_model_prediction_interf(self, image, file_name='networks/interf_trained.ckpt'):
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=[None, 28, 28])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        y_true = tf.placeholder(tf.float32, shape=[None, 10])

        convo_1 = convolutional_layer(x_image, shape=[6, 6, 1, 32])
        convo_1_pooling = max_pool_2by2(convo_1)
        convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 32, 64])
        convo_2_pooling = max_pool_2by2(convo_2)
        convo_2_flat = tf.reshape(convo_2_pooling, shape=[-1, 7 * 7 * 64])
        full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

        hold_prob = tf.placeholder(tf.float32, None)
        full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=1)
        y_pred = normal_full_layer(full_one_dropout, 10)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, file_name)
            image = np.reshape(image, newshape=[28, 28])
            # a = image
            a = np.reshape(image, newshape=[1, 28, 28])
            feed_dict = {x: a}
            classification = sess.run(y_pred, feed_dict)
            # classification[0][26]=-1000
            # classification[0][16] = -1000
            #print(classification)
            predictie = np.argmax(classification)
            #if (np.argmax(classification) < 11):
            #    print(np.argmax(classification))
            #    a = np.argmax(classification)
            #    print(a)
            #else:
            #    print(predictie)
        return classification


    def convo_model_prediction(self, image, file_name='networks/1chr_digits_model_conv.ckpt'):
        tf.reset_default_graph()
        x = tf.placeholder(tf.float32, shape=[None, 28, 28])
        x_image = tf.reshape(x, [-1, 28, 28, 1])
        y_true = tf.placeholder(tf.float32, shape=[None, 10])

        convo_1 = convolutional_layer(x_image, shape=[6, 6, 1, 32])
        convo_1_pooling = max_pool_2by2(convo_1)
        convo_2 = convolutional_layer(convo_1_pooling, shape=[5, 5, 32, 64])
        convo_2_pooling = max_pool_2by2(convo_2)
        convo_2_flat = tf.reshape(convo_2_pooling, shape=[-1, 7 * 7 * 64])
        full_layer_one = tf.nn.relu(normal_full_layer(convo_2_flat, 1024))

        hold_prob = tf.placeholder(tf.float32, None)
        full_one_dropout = tf.nn.dropout(full_layer_one, keep_prob=1)
        y_pred = normal_full_layer(full_one_dropout, 10)
        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
        train = optimizer.minimize(cross_entropy)
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(init)
            saver.restore(sess, file_name)
            image = np.reshape(image, newshape=[28, 28])
            # a = image
            a = np.reshape(image, newshape=[1, 28, 28])
            feed_dict = {x: a}
            classification = sess.run(y_pred, feed_dict)
            # classification[0][26]=-1000
            # classification[0][16] = -1000
            #print(classification)
            #predictie = np.argmax(classification)
            #if (np.argmax(classification) < 11):
            #    print(np.argmax(classification))
            #    a = np.argmax(classification)
            #    print(a)
            #else:
            #    print(predictie)
        return classification



