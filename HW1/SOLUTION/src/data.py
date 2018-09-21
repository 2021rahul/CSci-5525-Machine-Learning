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


class DATA():

    def __init__(self):
        self.dataX = None
        self.dataY = None
        self.classes = None
        self.splits = []
        self.index = 0

    def load_dataset(self):
        raise NotImplementedError

    def get_kfold_splits(self):
        kf = KFold(n_splits=config.n_splits)
        kf.get_n_splits(self.dataX)
        for train, test in kf.split(self.dataX):
            self.splits.append([train, test])

    def generate_data(self):
        if(self.index < config.n_splits):
            trainX = self.dataX[self.splits[self.index][0], :]
            trainY = self.dataY[self.splits[self.index][0], :]
            testX = self.dataX[self.splits[self.index][1], :]
            testY = self.dataX[self.splits[self.index][1], :]
            self.index = self.index + 1
            return trainX, trainY, testX, testY


class Boston(DATA):

    def __init__(self):
        super(Boston, self).__init__()

    def load_dataset(self):
        data = pd.read_csv(os.path.join(config.DATA_DIR, 'boston.csv'), header=None)
        self.dataX = data.iloc[:, :-1].values
        self.dataY = data.iloc[:, -1].values

    def categorize_data(self):
        threshold = np.percentile(self.dataY, config.THRESHOLD_PERCENTILE)
        self.dataY[self.dataY < threshold] = 0
        self.dataY[self.dataY >= threshold] = 1
        self.dataY = np.reshape(self.dataY, (len(self.dataY), 1))
        self.classes = np.unique(self.dataY)


class Digits(DATA):

    def __init__(self):
        super(Boston, self).__init__()

    def load_dataset(self):
        data = pd.read_csv(os.path.join(config.DATA_DIR, 'digits.csv'), header=None)
        self.dataX = data.iloc[:, :-1].values
        self.dataY = data.iloc[:, -1].values