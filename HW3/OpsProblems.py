#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 28 09:31:01 2018

@author: 2021rahul
"""

"""
Simple TensorFlow exercises
You should thoroughly test your code
"""

import tensorflow as tf
import numpy as np

###############################################################################
# 1a: Create two random 0-d tensors x and y that are Gaussian distributed, zero mean unit variance.
# Create a TensorFlow object that returns x + y if x > y, and x - y otherwise.
# Hint: look up tf.cond() and modify example code below
###############################################################################
print("--------------------------------------------------")
print("Question 1a")
x = tf.random_uniform([])
y = tf.random_uniform([])
out = tf.cond(tf.greater(x, y), lambda: tf.add(x, y), lambda: tf.subtract(x, y))
with tf.Session() as sess:
    x_val, y_val, output = sess.run((x, y, out))
    print("X_val: %f, Y_val: %f, Output: %f" %(x_val, y_val, output))
###############################################################################
# 1b: Create two 0-d tensors x and y randomly selected from the range [-1, 1).
# Return x + y if x < y, x - y if x > y, 0 otherwise.
# Hint: Look up tf.case().
###############################################################################
print("--------------------------------------------------")
print("Question 1b")
x = tf.random_uniform([], minval=-1, maxval=1)
y = tf.random_uniform([], minval=-1, maxval=1)
def f1() : return tf.add(x, y)
def f2() : return tf.subtract(x, y)
def f3() : return tf.constant(0.0)
pred_fn_pairs = {tf.less(x, y):f1, tf.greater(x, y):f2}
out = tf.case(pred_fn_pairs, default=f3)
with tf.Session() as sess:
    x_val, y_val, output = sess.run((x, y, out))
    print("X_val: %f, Y_val: %f, Output: %f" %(x_val, y_val, output))
###############################################################################
# 1c: Create the tensor x of the value [[0, -2, -1], [0, 1, 2]]
# and y as a tensor of zeros with the same shape as x.
# Return a boolean tensor that yields Trues if x equals y element-wise.
# Hint: Look up tf.equal().
###############################################################################
print("--------------------------------------------------")
print("Question 1c")
x = tf.constant([[0, -2, -1], [0, 1, 2]])
y = tf.zeros_like(x)
out = tf.equal(x, y)
with tf.Session() as sess:
    x_val, y_val, output = sess.run((x, y, out))
    print("x_val:",x_val)
    print(" y_val:",y_val)
    print(" Output:", output)
###############################################################################
# 1d: Create the tensor x of value
# [29.05088806,  27.61298943,  31.19073486,  29.35532951,
#  30.97266006,  26.67541885,  38.08450317,  20.74983215,
#  34.94445419,  34.45999146,  29.06485367,  36.01657104,
#  27.88236427,  20.56035233,  30.20379066,  29.51215172,
#  33.71149445,  28.59134293,  36.05556488,  28.66994858].
# Get the indices of elements in x whose values are greater than 30.
# Hint: Use tf.where().
# Then extract elements whose values are greater than 30.
# Hint: Use tf.gather().
###############################################################################
print("--------------------------------------------------")
print("Question 1d")
x = tf.constant([29.05088806,  27.61298943,  31.19073486,  29.35532951,
                 30.97266006,  26.67541885,  38.08450317,  20.74983215,
                 34.94445419,  34.45999146,  29.06485367,  36.01657104,
                 27.88236427,  20.56035233,  30.20379066,  29.51215172,
                 33.71149445,  28.59134293,  36.05556488,  28.66994858])
indices = tf.where(tf.greater(x, 30.0))
vals = tf.gather(x, indices)
with tf.Session() as sess:
    output = sess.run((vals))
    print("Output:", output)
###############################################################################
# 1e: Create a diagnoal 2-d tensor of size 6 x 6 with the diagonal values of 1,
# 2, ..., 6
# Hint: Use tf.range() and tf.diag().
###############################################################################
print("--------------------------------------------------")
print("Question 1d")
x = tf.diag(tf.range(1,7))
with tf.Session() as sess:
    x_val = sess.run((x))
    print("X_val:", x_val)
###############################################################################
# 1f: Create a random 2-d tensor of size 10 x 10 from any distribution.
# Calculate its determinant.
# Hint: Look at tf.matrix_determinant().
###############################################################################
print("--------------------------------------------------")
print("Question 1f")
x = tf.random_uniform(shape=(10,10))
out = tf.matrix_determinant(x)
with tf.Session() as sess:
    x_val, output = sess.run((x, out))
    print("x_val:",x_val)
    print(" Output:", output)
###############################################################################
# 1g: Create tensor x with value [5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9].
# Return the unique elements in x
# Hint: use tf.unique(). Keep in mind that tf.unique() returns a tuple.
###############################################################################
print("--------------------------------------------------")
print("Question 1g")
x = tf.constant([5, 2, 3, 5, 10, 6, 2, 3, 4, 2, 1, 1, 0, 9])
out = tf.unique(x)
with tf.Session() as sess:
    x_val, output = sess.run((x, out))
    print("x_val:",x_val)
    print(" Output:", output[0])
###############################################################################
# 1h: Create two tensors x and y of shape 300 from any normal distribution,
# as long as they are from the same distribution. Compute
# - The mean squared error of (x - y) 
###############################################################################
print("--------------------------------------------------")
print("Question 1h")
x = tf.random_uniform([300])
y = tf.random_uniform([300])
out = tf.reduce_mean(tf.squared_difference(x, y))
init_g = tf.global_variables_initializer()
init_l = tf.local_variables_initializer()
with tf.Session() as sess:
    sess.run(init_g)
    sess.run(init_l)
    x_val, y_val, output = sess.run((x, y, out))
    print(" Output:", output)