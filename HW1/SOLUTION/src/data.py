#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:20:43 2018

@author: 2021rahul
"""

import os
import numpy as np
import pandas as pd
import config
import random
import utils
from sklearn.preprocessing import OneHotEncoder


class DATA():

    def __init__(self):
        self.dataX = None
        self.dataY = None
        self.classes = None
        self.index = 0

    def load_dataset(self):
        raise NotImplementedError

    def one_hot_encoded(self):
        onehot_encoder = OneHotEncoder(sparse=False)
        self.dataY = onehot_encoder.fit_transform(self.dataY.reshape(-1, 1))

    def generate_data(self):
        Y = self.dataY.argmax(axis=1)
        Y = np.reshape(Y, (-1, 1))

        grouped_data = utils.group_data(self.dataX, Y)
        grouped_dataY = utils.group_data(self.dataY, Y)

        trainX = testX = np.zeros((1, self.dataX.shape[1]))
        trainY = testY = np.zeros((1, self.dataY.shape[1]))

        for key in grouped_data:
            dataX = grouped_data[key]
            dataY = grouped_dataY[key]

            index = random.sample(range(0, len(dataX)), len(dataX))
            split_index = int(0.8*len(dataX))

            trainX = np.concatenate((trainX, dataX[index[:split_index], :]))
            testX = np.concatenate((testX, dataX[index[split_index:], :]))
            trainY = np.concatenate((trainY, dataY[index[:split_index], :]))
            testY = np.concatenate((testY, dataY[index[split_index:], :]))

        trainX, trainY = utils.randomize_data(trainX[1:, :], trainY[1:, :])
        testX, testY = utils.randomize_data(testX[1:, :], testY[1:, :])

        return trainX, trainY, testX, testY


class Boston(DATA):

    def __init__(self):
        super(Boston, self).__init__()

    def load_dataset(self):
        data = pd.read_csv(os.path.join(config.DATA_DIR, 'boston.csv'), header=None)
        self.dataX = data.iloc[:, :-1].values
        self.dataY = data.iloc[:, -1].values
        self.dataY = np.reshape(self.dataY, (-1, 1))

    def categorize_data(self):
        threshold = np.percentile(self.dataY, 50)
        self.dataY[self.dataY < threshold] = 0
        self.dataY[self.dataY >= threshold] = 1


class Digits(DATA):

    def __init__(self):
        super(Digits, self).__init__()

    def load_dataset(self):
        data = pd.read_csv(os.path.join(config.DATA_DIR, 'digits.csv'), header=None)
        self.dataX = data.iloc[:, :-1].values
        self.dataY = data.iloc[:, -1].values
        self.dataY = np.reshape(self.dataY, (-1, 1))

dataset = Boston()
dataset.load_dataset()
dataset.categorize_data()
dataset.one_hot_encoded()
x, y, tx, ty = dataset.generate_data()

