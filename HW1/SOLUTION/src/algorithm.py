#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:20:43 2018

@author: 2021rahul
"""


import utils
import numpy as np
import math


class LDA():

    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.eigens = None

    def get_projections(self, data):
        Sb = utils.between_class_scatter(data.dataX, data.dataY)
        Sw = utils.within_class_scatter(data.dataX, data.dataY)
        mat = np.dot(np.linalg.inv(Sw), Sb)
        self.eigens = utils.get_sorted_eigens(mat)
        self.eigens = np.reshape(self.eigens[self.dimensions, 1:], (len(self.eigens), self.dimensions))


class LogisticRegression():

    def __init__(self, shape):
        self.weights = np.zeros((shape[0], shape[1]))
        self.bias = np.zeros(shape[1])

    def forward_propogation(self, x):
        return utils.softmax(self.weights, self.bias, x)

    def negative_log_likelihood(self, x, y):
        sigmoid = self.forward_propogation(x)
        log_likelihood = - np.mean(np.sum(y * np.log(sigmoid) + (1 - y) * np.log(1 - sigmoid), axis=1))
        return log_likelihood

    def gradient_update(self, x, y, lr_rate):
        sigmoid = self.forward_propogation(x)
        e = y-sigmoid
        self.weights += lr_rate * np.dot(x.T, e)
        self.bias += lr_rate * np.mean(e, axis=0)

    def train(self, dataX, dataY, lr_rate, n_epochs):
        for epoch in range(n_epochs):
            self.gradient_update(dataX, dataY, lr_rate)
            log_likelihood = self.negative_log_likelihood(dataX, dataY)
#            print("Epoch: ", epoch, " error: ", log_likelihood)

    def test(self, dataX, dataY):
        prediction = self.forward_propogation(dataX)
        prediction = prediction.argmax(axis=1)
        target = dataY.argmax(axis=1)
        count = np.sum(prediction == target)
#        print("Accuracy: ", count/len(target))
        return count


class NaiveBayes():

    def __init__(self):
        self.class_mean = {}
        self.class_stddev = {}

    def separate_data_into_classes(self, dataX, dataY):
        separate_data = {}
        classes = np.unique(dataY)
        for class_val in classes:
            get_data = dataX[dataY == class_val, :]
            separate_data[class_val] = get_data
        return separate_data

    def calulate_classwise_summary(self, dataX, dataY):
        separate_data = self.separate_data_into_classes(dataX, dataY)
        for key, data in separate_data.items():
            self.class_mean[key] = utils.mean(data)
            self.class_stddev[key] = utils.stddev(data)

    def calculate_probability(self, x, mean, stddev):
        exp_num = np.power(x-mean, 2)
        exp_den = 2*np.power(stddev, 2)
        num = np.exp(np.divide(-exp_num, exp_den))
        den = np.sqrt(2*math.pi*stddev)
        vals = np.divide(num, den)
        vals[np.isnan(vals)] = 1
        return np.prod(vals)

    def train(self, trainX, trainY):
        self.calulate_classwise_summary(trainX, trainY)

    def predict(self, probabilities):
        best_label = None
        best_val = None
        for class_val, probability in probabilities.items():
            if best_label is None or probability > best_val:
                best_label = class_val
                best_val = probability
        return best_label

    def test(self, testX, testY):
        predictions = []
        for data in testX:
            probabilities = {}
            for class_val, _ in self.class_mean.items():
                prob = self.calculate_probability(data, self.class_mean[class_val], self.class_stddev[class_val])
                probabilities[class_val] = prob
            predictions.append(self.predict(probabilities))
        predictions = np.asarray(predictions)
        count = np.sum(predictions == testY)
        return count



















