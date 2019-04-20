# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import cv2
import os
import random

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

train_data_dir = '/scratch/user/caoqing2017/Verfication_Code/trainImage'
val_data_dir = '/scratch/user/caoqing2017/Verfication_Code/validateImage'
width = 160
height = 60
max_captcha = 4
batch_size = 128
num_char = len(number)

def get_next_batch(data_dir):

    simples = {}
    for file_name in os.listdir(data_dir):
        captcha = file_name.split('.')[0]
        simples[data_dir + '/' + file_name] = captcha

    file_simples = list(simples.keys())
    num_simples = len(simples)

    batch_x = np.zeros([batch_size, width * height])
    batch_y = np.zeros([batch_size, num_char * max_captcha])

    for i in range(batch_size):
        file_name = file_simples[random.randint(0, num_simples - 1)]
        batch_x[i, :] = np.float32(cv2.imread(file_name, 0)).flatten() / 255
        batch_y[i, :] = text2vec(simples[file_name])
    return batch_x, batch_y


def text2vec(text):

    return [0 if ord(i) - 48 != j else 1 for i in text for j in range(num_char)]


x = tf.placeholder(tf.float32, [None, width * height], name='input')
y_ = tf.placeholder(tf.float32, [None, num_char * max_captcha])
x_image = tf.reshape(x, [-1, height, width, 1])
keep_prob = tf.placeholder(tf.float32, name='dropout')


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_pool1 = tf.nn.dropout(h_pool1, keep_prob)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2 = tf.nn.dropout(h_pool2, keep_prob)

W_conv3 = weight_variable([5, 5, 64, 64])
b_conv3 = bias_variable([64])
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)
h_pool3 = tf.nn.dropout(h_pool3, keep_prob)

W_fc1 = weight_variable([8 * 20 * 64, 1024])
b_fc1 = bias_variable([1024])
h_pool3_flat = tf.reshape(h_pool3, [-1, 8 * 20 * 64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)

W_fc2 = weight_variable([1024, num_char * max_captcha])
b_fc2 = bias_variable([num_char * max_captcha])
output = tf.add(tf.matmul(h_fc1, W_fc2), b_fc2)

cross_entropy = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_, logits=output))
train_step = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cross_entropy)

predict = tf.reshape(output, [-1, max_captcha, num_char])
labels = tf.reshape(y_, [-1, max_captcha, num_char])
correct_prediction = tf.equal(tf.argmax(predict, 2, name='predict_max_idx'), tf.argmax(labels, 2))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


def train():
    sess = tf.InteractiveSession()

    saver = tf.train.Saver()
    tf.global_variables_initializer().run()

    for i in range(90000):
        batch_x, batch_y = get_next_batch(train_data_dir)
        vali_batch_x, vali_batch_y = get_next_batch(val_data_dir)

        feed_dict_tr = {x: batch_x, y_: batch_y, keep_prob: 0.8}
        feed_dict_val = {x: vali_batch_x, y_: vali_batch_y, keep_prob: 0.8}

        train_step.run(feed_dict=feed_dict_tr)

        if i % 1000 == 0:
            train_accuracy = accuracy.eval(feed_dict=feed_dict_tr)
            train_loss = cross_entropy.eval(feed_dict=feed_dict_tr)
            vali_accuracy = accuracy.eval(feed_dict=feed_dict_val)
            print("step {}, training accuracy {} , validation accuracy {}, training loss {}" .format(i, train_accuracy, vali_accuracy, train_loss))

            if train_accuracy > 0.9:
                saver.save(sess, '/scratch/user/caoqing2017/Verfication_Code/output.model', global_step=i)


train()
