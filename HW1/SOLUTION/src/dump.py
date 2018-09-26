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
import math


dataset = data.Boston()
dataset.load_dataset()
dataset.get_kfold_splits(10)
trainX, trainY, testX, testY = dataset.generate_data()
trainY = trainY.argmax(axis=1)
testY = testY.argmax(axis=1)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components = 1)
see = lda.fit_transform(trainX, trainY)
trainY = np.asarray(np.reshape(trainY, (len(trainY), 1)))
grouped_data = utils.group_data(trainX, trainY)

#utils.plot_histograms(see, np.reshape(trainY, (len(trainY), 1)))



#seperate_data = {}
#classes = np.unique(trainY)
#for class_val in classes:
#    get_data = trainX[trainY == class_val, :]
#    seperate_data[class_val] = get_data
#
#class_summaries = {}
#for key, data in seperate_data.items():
#    summaries = {}
#    summaries["mean"] = utils.mean(data)
#    summaries["stddev"] = utils.stddev(data)
#    class_summaries[key] = summaries
#
#def calculate_probability(x, mean, stddev):
#    exp_num = np.power(x-mean, 2)
#    exp_den = 2*np.power(stddev, 2)
#    num = np.exp(np.divide(-exp_num, exp_den))
#    den = np.sqrt(2*math.pi*stddev)
#    vals = np.divide(num, den)
#    vals[np.isnan(vals)] = 1
#    return np.prod(vals)
#
#predictions = []
#for data in testX:
#    probabilities = {}
#    for class_val, class_summary in class_summaries.items():
#        mean = class_summary["mean"]
#        stddev = class_summary["stddev"]
#        prob = calculate_probability(data, mean, stddev)
#        probabilities[class_val] = prob
#    probability = 0
#    best_label = None
#    best_val = None
#    for class_val, probability in probabilities.items():
#        if best_label is None or probability > best_val:
#            best_label = class_val
#            best_val = probability
#    predictions.append(best_label)
#
#predictions = np.asarray(predictions)
#accuracy = np.sum(predictions==testY)

#X = dataset.dataX
#Y = dataset.dataY
#weights = np.zeros((64, 10))
#bias = np.zeros(10)
#
#kf = KFold(n_splits=num_splits)
#kf.get_n_splits(self.dataX)
#
#def softmax(weights, bias, x):
#    output = np.dot(x, weights)
#    output = np.add(output, bias)
#    output = np.exp(output)
#    output = output.T/np.sum(output, axis=1)
#    return output.T
#
#
#def forward_propogation(x):
#    return softmax(weights, bias, x)
#
#
#def negative_log_likelihood(x, y):
#    sigmoid = forward_propogation(x)
#    log_likelihood = - np.mean(np.sum(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid), axis=1))
#    return log_likelihood
#
#
#def gradient_update(weights, bias, x, y, lr_rate):
#    sigmoid = forward_propogation(x)
#    e = y-sigmoid
#    weights += lr_rate * np.dot(x.T, e)
#    bias += lr_rate * np.mean(e, axis=0)
#
#
#for i in range(10000):
#    gradient_update(weights, bias, X, Y, 0.00001)
#    log_likelihood = negative_log_likelihood(X, Y)
#    see = forward_propogation(X)
#    print(log_likelihood)
#
#pred_indices = see.argmax(axis=1)
#true_indices = Y.argmax(axis=1)
#
#accuracy = np.sum(pred_indices==true_indices)


def word_count(text):
    print(len(text.split(" ")))
