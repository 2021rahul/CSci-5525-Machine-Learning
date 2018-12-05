#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import division

"""
Created on Wed Nov 28 09:32:47 2018

@author: 2021rahul
"""

""" Using convolutional net on MNIST dataset of handwritten digit
(http://yann.lecun.com/exdb/mnist/)
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

N_CLASSES = 10

# Step 1: Read in data
# using TF Learn's built in function to load MNIST data to the folder data/mnist
mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

# Step 2: Define paramaters for the model
LEARNING_RATE = 0.001
BATCH_SIZE = 128
SKIP_STEP = 10
DROPOUT = 0.75
N_EPOCHS = 1

# Step 3: create placeholders for features and labels
# each image in the MNIST data is of shape 28*28 = 784
# therefore, each image is represented with a 1x784 tensor
# We'll be doing dropout for hidden layer so we'll need a placeholder
# for the dropout probability too
# Use None for shape so we can change the batch_size once we've built the graph
with tf.name_scope('data'):
    X = tf.placeholder(tf.float32, [None, 784], name="X_placeholder")
    Y = tf.placeholder(tf.float32, [None, 10], name="Y_placeholder")

dropout = tf.placeholder(tf.float32, name='dropout')

# Step 4 + 5: create weights + do inference
# the model is conv -> relu -> pool -> conv -> relu -> pool -> fully connected -> softmax

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

CHECKPOINT_DIR = 'checkpoints/'
if not os.path.exists(CHECKPOINT_DIR):
    os.makedirs(CHECKPOINT_DIR)
MNIST_DIR = 'checkpoints/convnet_mnist'
if not os.path.exists(MNIST_DIR):
    os.makedirs(MNIST_DIR)

with tf.variable_scope('conv1') as scope:
    images = tf.reshape(X, shape=[-1, 28, 28, 1])
    kernel1 = tf.get_variable('kernels', [5, 5, 1, 32],
                             initializer=tf.truncated_normal_initializer())
    biases1 = tf.get_variable('biases', [32],
                             initializer=tf.constant_initializer())
    conv = tf.nn.conv2d(images, kernel1, strides=[1, 1, 1, 1], padding='SAME')
    conv1 = tf.nn.relu(conv + biases1, name=scope.name)

with tf.variable_scope('pool1') as scope:
# apply max pool with ksize [1, 2, 2, 1], and strides [1, 2, 2, 1], padding 'SAME'

    pool1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

# output is of dimension BATCH_SIZE x 14 x 14 x 32

with tf.variable_scope('conv2') as scope:
    # similar to conv1, except kernel now is of the size 5 x 5 x 32 x 64
    kernel2 = tf.get_variable('kernels', [5, 5, 32, 64],
                             initializer=tf.truncated_normal_initializer())
    biases2 = tf.get_variable('biases', [64],
                             initializer=tf.random_normal_initializer())
    conv = tf.nn.conv2d(pool1, kernel2, strides=[1, 1, 1, 1], padding='SAME')
    conv2 = tf.nn.relu(conv + biases2, name=scope.name)

    # output is of dimension BATCH_SIZE x 14 x 14 x 64

with tf.variable_scope('pool2') as scope:
    # similar to pool1
    pool2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],
                           padding='SAME')

    # output is of dimension BATCH_SIZE x 7 x 7 x 64

with tf.variable_scope('fc') as scope:
    # use weight of dimension 7 * 7 * 64 x 1024
    input_features = 7 * 7 * 64

    # create weights and biases

    weights_fc = tf.get_variable('weights', [input_features, 1024],
                             initializer=tf.truncated_normal_initializer())
    biases_fc = tf.get_variable('biases', [1024],
                             initializer=tf.random_normal_initializer())
    # reshape pool2 to 2 dimensional
    pool2 = tf.reshape(pool2, [-1, input_features])

    # apply relu on matmul of pool2 and w + b
    fc = tf.nn.relu(tf.matmul(pool2, weights_fc) + biases_fc, name='relu')

    # apply dropout
    fc = tf.nn.dropout(fc, dropout, name='relu_dropout')

with tf.variable_scope('softmax_linear') as scope:
# this you should know. get logits without softmax
# you need to create weights and biases
    weights_softmax = tf.get_variable('weights', [1024, 10],
                             initializer=tf.truncated_normal_initializer())
    biases_softmax = tf.get_variable('biases', [10],
                             initializer=tf.random_normal_initializer())
    logits = tf.nn.bias_add(tf.matmul(fc, weights_softmax, name="multiply_weights"), biases_softmax, name="add_bias")

# Step 6: define loss function
# use softmax cross entropy with logits as the loss function
# compute mean cross entropy, softmax is applied internally
with tf.name_scope('loss'):
# you should know how to do this too
   loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
# TO DO
# Step 7: define training op
# using gradient descent with learning rate of LEARNING_RATE to minimize cost
# don't forgot to pass in global_step
with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss, global_step)

# TO DO

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    # to visualize using TensorBoard
    writer = tf.summary.FileWriter('./my_graph/mnist', sess.graph)
    ##### You have to create folders to store checkpoints
    ckpt = tf.train.get_checkpoint_state(os.path.dirname('checkpoints/convnet_mnist/checkpoint'))
    # if that checkpoint exists, restore from checkpoint
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)

    initial_step = global_step.eval()

    start_time = time.time()
    n_batches = int(mnist.train.num_examples / BATCH_SIZE)

    total_loss = 0.0
    for index in range(initial_step, n_batches * N_EPOCHS):  # train the model n_epochs times
        X_batch, Y_batch = mnist.train.next_batch(BATCH_SIZE)
        _, loss_batch = sess.run([optimizer, loss],
                                 feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT})
        total_loss += loss_batch
        if (index + 1) % SKIP_STEP == 0:
            print('Average loss at step {}: {:5.1f}'.format(index + 1, total_loss / SKIP_STEP))
            total_loss = 0.0
            saver.save(sess, 'checkpoints/convnet_mnist/mnist-convnet', index)

    print("Optimization Finished!")  # should be around 0.35 after 25 epochs
    print("Total time: {0} seconds".format(time.time() - start_time))

    # test the model
    n_batches = int(mnist.test.num_examples / BATCH_SIZE)
    total_correct_preds = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(BATCH_SIZE)
        _, loss_batch, logits_batch = sess.run([optimizer, loss, logits],
                                               feed_dict={X: X_batch, Y: Y_batch, dropout: DROPOUT})
        preds = tf.nn.softmax(logits_batch)
        correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y_batch, 1))
        accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32))
        total_correct_preds += sess.run(accuracy)

    print("Accuracy {0}".format(total_correct_preds / mnist.test.num_examples))