#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:20:43 2018

@author: 2021rahul
"""

import os
import numpy as np
import pandas as pd
import collections
import config
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder


class DATA():

    def __init__(self):
        self.dataX = None
        self.dataY = None
        self.classes = None
        self.splits = []
        self.num_splits = None
        self.index = 0

    def load_dataset(self):
        raise NotImplementedError

    def one_hot_encoded(self):
        onehot_encoder = OneHotEncoder(sparse=False)
        self.dataY = onehot_encoder.fit_transform(self.dataY.reshape(-1, 1))

    def get_kfold_splits(self, num_splits=10):
        self.num_splits = num_splits
        kf = KFold(n_splits=num_splits)
        kf.get_n_splits(self.dataX)
        for train, test in kf.split(self.dataX):
            self.splits.append([train, test])

    def generate_data(self):
        if(self.index < self.num_splits):
            trainX = self.dataX[self.splits[self.index][0], :]
            trainY = self.dataY[self.splits[self.index][0], :]
            testX = self.dataX[self.splits[self.index][1], :]
            testY = self.dataY[self.splits[self.index][1], :]
            self.index = self.index + 1
            return trainX, trainY, testX, testY


class Boston(DATA):

    def __init__(self):
        super(Boston, self).__init__()

    def load_dataset(self):
        data = pd.read_csv(os.path.join(config.DATA_DIR, 'boston.csv'), header=None)
        self.dataX = data.iloc[:, :-1].values
        self.dataY = data.iloc[:, -1].values
        self.dataY = np.reshape(self.dataY, (-1,1))

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
        self.dataY = np.reshape(self.dataY, (-1,1))

