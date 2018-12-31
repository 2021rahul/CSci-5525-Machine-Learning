#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 09:32:47 2018

@author: 2021rahul
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

learning_rate=0.0001
n_classes=10
batch_size=128
n_epochs = 25
skip_step = 10
dropout = 0.75

with tf.name_scope('data'):
    X = tf.placeholder(shape=[None, 784], dtype=tf.float32, name="inputs")
    Y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name="labels")
DROPOUT = tf.placeholder(tf.float32, name='DROPOUT')

images = tf.reshape(X, shape=[-1, 28, 28, 1])

with tf.variable_scope('conv1') as scope:
    kernel1 = tf.get_variable('kernels', [5, 5, 1, 32],
                             initializer=tf.truncated_normal_initializer())
    biases1 = tf.get_variable('biases', [32],
                             initializer=tf.constant_initializer())
    conv = tf.nn.conv2d(images, kernel1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv + biases1, name=scope.name)

with tf.variable_scope('pool1') as scope:
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

with tf.variable_scope('conv2') as scope:
    kernel2 = tf.get_variable('kernels', [5, 5, 32, 64],
                             initializer=tf.truncated_normal_initializer())
    biases2 = tf.get_variable('biases', [64],
                             initializer=tf.constant_initializer())
    conv = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv + biases2, name=scope.name)

with tf.variable_scope('pool2') as scope:
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

input_features = 7 * 7 * 64
pool2 = tf.reshape(pool2, [-1, input_features])

with tf.variable_scope('fc') as scope:
    weights_fc = tf.get_variable('weights', [input_features, 1024],
                             initializer=tf.truncated_normal_initializer())
    biases_fc = tf.get_variable('biases', [1024],
                             initializer=tf.constant_initializer())

    fc = tf.nn.relu(tf.matmul(pool2, weights_fc) + biases_fc, name='relu')
    fc = tf.nn.dropout(fc, DROPOUT, name='relu_dropout')

with tf.variable_scope('softmax_linear') as scope:
    weights_softmax = tf.get_variable('weights', [1024, 10],
                             initializer=tf.truncated_normal_initializer())
    biases_softmax = tf.get_variable('biases', [10],
                             initializer=tf.constant_initializer())
    logits = tf.nn.bias_add(tf.matmul(fc, weights_softmax, name="multiply_weights"), biases_softmax, name="add_bias")

with tf.name_scope("cost_function"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step)

merged_summary_op = tf.summary.merge_all()

with tf.Session() as sess:    
    start_time = time.time()
    summary_writer = tf.summary.FileWriter("output/ConvNet", sess.graph)
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples / batch_size)

    for i in range(n_epochs):
        total_loss = 0
        for batch in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size)
            feed_dict = {X: X_batch, Y: Y_batch, DROPOUT: dropout}
            summary_str, _, loss_batch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=global_step.eval())
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
    
    print("Optimization Finished!")
    print("Total time: {0} seconds".format(time.time() - start_time))

    # test the model
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) 
    
    n_batches = int(mnist.test.num_examples / batch_size)
    accuracy_batch = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        feed_dict = {X: X_batch, Y: Y_batch, DROPOUT: dropout}
        accuracy_batch += sess.run(accuracy, feed_dict=feed_dict)

    print('Accuracy {0}'.format(accuracy_batch / mnist.test.num_examples))
