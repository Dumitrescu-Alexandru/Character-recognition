import numpy as np
import tensorflow as tf
from convo_functions import convolutional_layer
from convo_functions import max_pool_2by2
from convo_functions import normal_full_layer
def train_network(imgs,lbls,train_coeff,steps=10,file_name='networks/interf_trained.ckpt'):

    oh_lbls = []
    for lbl in lbls:
        current_label = np.zeros(10)
        current_label[lbl] = 1
        oh_lbls.append(current_label)
    oh_lbls = np.array(oh_lbls)
    lbls = oh_lbls
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
    optimizer = tf.train.AdamOptimizer(learning_rate=train_coeff)
    train = optimizer.minimize(cross_entropy)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        saver.restore(sess, file_name)
        for step in range(steps):
            for a in range(np.shape(imgs)[0]):
                img = imgs[a]
                lbl = lbls[a]
                img = np.reshape(img,(1,28,28))
                lbl = np.reshape(lbl,(1,10))
                batch_x = img
                batch_y = lbl
                sess.run(train, feed_dict={x: batch_x, y_true: batch_y, hold_prob: 1})
        saver.save(sess, file_name)

