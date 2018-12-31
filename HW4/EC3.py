#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 16 04:20:10 2018

@author: 2021rahul
"""
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

learning_rate=0.001

n_epochs = 25
NUM_STEPS = 28
NUM_UNITS =128
NUM_CLASSES =10

with tf.name_scope('data'):
    X = tf.placeholder(shape=[NUM_STEPS, 784], dtype=tf.float32, name="inputs")
    Y = tf.placeholder(shape=[NUM_STEPS, 10], dtype=tf.float32, name="labels")
    lstmY = tf.placeholder(shape=[NUM_STEPS-1, 10], dtype=tf.float32, name="lstm_labels")

with tf.name_scope('CNN') as scope:
    with tf.variable_scope('conv1') as scope:
        images = tf.reshape(X, shape=[-1, 28, 28, 1])
        kernel1 = tf.get_variable('kernels', [5, 5, 1, 6],
                                 initializer=tf.truncated_normal_initializer())
        biases1 = tf.get_variable('biases', [6],
                                 initializer=tf.constant_initializer())
        conv = tf.nn.conv2d(images, kernel1, strides=[1, 1, 1, 1], padding='SAME')
        conv1 = tf.nn.relu(conv + biases1, name=scope.name)
    
    with tf.variable_scope('pool1') as scope:
        pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
    
    with tf.variable_scope('conv2') as scope:
        kernel2 = tf.get_variable('kernels', [3, 3, 6, 16],
                                 initializer=tf.truncated_normal_initializer())
        biases2 = tf.get_variable('biases', [16],
                                 initializer=tf.constant_initializer())
        conv = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
        conv2 = tf.nn.relu(conv + biases2, name=scope.name)
    
    with tf.variable_scope('pool2') as scope:
        pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                               padding='SAME')
    
    with tf.variable_scope('fc') as scope:
        input_features = 7 * 7 * 16
        pool2 = tf.reshape(pool2, [-1, input_features])
        weights_fc = tf.get_variable('weights', [input_features, 128],
                                 initializer=tf.truncated_normal_initializer())
        biases_fc = tf.get_variable('biases', [128],
                                 initializer=tf.constant_initializer())
        fc = tf.nn.relu(tf.matmul(pool2, weights_fc) + biases_fc, name='relu')

#with tf.variable_scope('softmax_linear') as scope:
#    weights_softmax = tf.get_variable('weights', [128, 10],
#                             initializer=tf.truncated_normal_initializer())
#    biases_softmax = tf.get_variable('biases', [10],
#                             initializer=tf.constant_initializer())
#    cnn_logits = tf.nn.bias_add(tf.matmul(fc, weights_softmax, name="multiply_weights"), biases_softmax, name="add_bias")
#
#with tf.name_scope("cost_function"):
#    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=cnn_logits, labels=Y))
#tf.summary.scalar('loss', loss)
#
#global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
#with tf.name_scope("train"):
#    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
#
#with tf.Session() as sess:
#    summary_writer = tf.summary.FileWriter("output/CNN", sess.graph)

with tf.name_scope("LSTM"):
    lstm_input = tf.reshape(fc, (1, NUM_STEPS, NUM_UNITS))
    lstm_input=tf.unstack(lstm_input ,NUM_STEPS,1)
    lstm_layer=rnn.BasicLSTMCell(NUM_UNITS, forget_bias=1)
    out,_=rnn.static_rnn(lstm_layer, lstm_input, dtype="float32")
    outputs = []
    for output_ in out:
        outputs.append(tf.reshape(output_, [NUM_UNITS]))
    outputs = tf.stack(outputs[:-1])

with tf.variable_scope('softmax_linear') as scope:
    weights_softmax = tf.get_variable('weights', [NUM_UNITS, NUM_CLASSES],
                             initializer=tf.truncated_normal_initializer())
    biases_softmax = tf.get_variable('biases', [NUM_CLASSES],
                             initializer=tf.constant_initializer())
    lstm_logits = tf.nn.bias_add(tf.matmul(outputs, weights_softmax, name="multiply_weights"), biases_softmax, name="add_bias")


with tf.name_scope("cost_function"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=lstm_logits, labels=lstmY))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)
#
with tf.Session() as sess:
    summary_writer = tf.summary.FileWriter("output/CNN_LSTM", sess.graph)
