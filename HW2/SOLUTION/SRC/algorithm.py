#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:20:43 2018

@author: 2021rahul
"""


import utils
import numpy as np
import math


class SVM():

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

    def test(self, dataX, dataY):
        prediction = self.forward_propogation(dataX)
        prediction = prediction.argmax(axis=1)
        target = dataY.argmax(axis=1)
        count = np.sum(prediction == target)
        return count
