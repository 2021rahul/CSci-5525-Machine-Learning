#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 10:11:10 2018

@author: ghosh128
"""

import data
import config
import algorithm
import numpy as np
import utils
from sklearn.linear_model import LogisticRegression

dataset = data.Digits()
dataset.load_dataset()
dataset.get_kfold_splits(10)
trainX, trainY, testX, testY = dataset.generate_data()
trainY = trainY.argmax(axis=1)
testY = testY.argmax(axis=1)
LR = LogisticRegression()
LR.fit(trainX, trainY)
predict = LR.predict(testX)
accuracy = np.sum(predict==testY)


X = dataset.dataX
Y = dataset.dataY
weights = np.zeros((64, 10))
bias = np.zeros(10)

kf = KFold(n_splits=num_splits)
kf.get_n_splits(self.dataX)

def softmax(weights, bias, x):
    output = np.dot(x, weights)
    output = np.add(output, bias)
    output = np.exp(output)
    output = output.T/np.sum(output, axis=1)
    return output.T


def forward_propogation(x):
    return softmax(weights, bias, x)


def negative_log_likelihood(x, y):
    sigmoid = forward_propogation(x)
    log_likelihood = - np.mean(np.sum(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid), axis=1))
    return log_likelihood


def gradient_update(weights, bias, x, y, lr_rate):
    sigmoid = forward_propogation(x)
    e = y-sigmoid
    weights += lr_rate * np.dot(x.T, e)
    bias += lr_rate * np.mean(e, axis=0)


for i in range(10000):
    gradient_update(weights, bias, X, Y, 0.00001)
    log_likelihood = negative_log_likelihood(X, Y)
    see = forward_propogation(X)
    print(log_likelihood)

pred_indices = see.argmax(axis=1)
true_indices = Y.argmax(axis=1)

accuracy = np.sum(pred_indices==true_indices)
