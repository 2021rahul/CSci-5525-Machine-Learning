#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 30 15:51:13 2018

@author: 2021rahul
"""

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import time

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn


mnist = input_data.read_data_sets("/data/mnist", one_hot=True)

time_steps=28
num_units=128
n_input=28
learning_rate=0.001
n_classes=10
batch_size=128
n_epochs = 25


with tf.name_scope('data'):
    X = tf.placeholder(shape=[None, 784], dtype=tf.float32, name="inputs")
    Y = tf.placeholder(shape=[None, 10], dtype=tf.float32, name="labels")

print(X.get_shape())
images = tf.reshape(X, (batch_size,time_steps,n_input))    
print(images.get_shape())
images=tf.unstack(images ,time_steps,1)
print(images.get_shape())

with tf.name_scope("LSTM"):
    lstm_layer=rnn.BasicLSTMCell(num_units, forget_bias=1)
    outputs,_=rnn.static_rnn(lstm_layer, images, dtype="float32")

with tf.name_scope("Variables"):
    with tf.name_scope("Weights"):
        weights = tf.Variable(tf.truncated_normal(shape=[num_units, n_classes], stddev=0.1))
    with tf.name_scope("Biases"):
        biases = tf.Variable(tf.constant(value=0.1, shape=[n_classes]))

logits = tf.nn.bias_add(tf.matmul(outputs[-1], weights, name="multiply_weights"), biases, name="add_bias")

with tf.name_scope("cost_function"):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
tf.summary.scalar('loss', loss)

global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step)


merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    start_time = time.time()
    summary_writer = tf.summary.FileWriter("output/vanillaLSTM", sess.graph)
    sess.run(tf.global_variables_initializer())
    n_batches = int(mnist.train.num_examples / batch_size)
    
    for i in range(n_epochs):
        total_loss = 0
        for batch in range(n_batches):
            X_batch, Y_batch = mnist.train.next_batch(batch_size=batch_size)
            feed_dict = {X: X_batch, Y: Y_batch}
            summary_str, _, loss_batch = sess.run([merged_summary_op, optimizer, loss], feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, global_step=global_step.eval())
            total_loss += loss_batch
        print('Average loss epoch {0}: {1}'.format(i, total_loss / n_batches))
       
    print("Optimization Finished!")
    print("Total time: {0} seconds".format(time.time() - start_time))

    #calculating test accuracy
    preds = tf.nn.softmax(logits)
    correct_preds = tf.equal(tf.argmax(preds, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_sum(tf.cast(correct_preds, tf.float32)) 
    
    n_batches = int(mnist.test.num_examples / batch_size)
    accuracy_batch = 0
    for i in range(n_batches):
        X_batch, Y_batch = mnist.test.next_batch(batch_size)
        accuracy_batch += sess.run(accuracy, feed_dict={X: X_batch, Y: Y_batch})

    print("Accuracy {0}".format(accuracy_batch/mnist.test.num_examples))
