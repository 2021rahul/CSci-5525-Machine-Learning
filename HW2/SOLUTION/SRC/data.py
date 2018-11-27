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
import utils
from sklearn.preprocessing import normalize
import random


class DATA():

    def __init__(self):
        self.dataX = None
        self.dataY = None
        self.classes = None
        self.index = 0

    def read_data(self, filename):
        data = pd.read_csv(os.path.join(config.DATA_DIR, filename), header=None)
        self.dataX = data.iloc[:, 1:].values
        self.dataX = normalize(self.dataX)
        self.dataY = np.reshape(data.iloc[:,0].values, (-1, 1))
        self.dataY[self.dataY == 3] = -1

    def generate_data(self):
        grouped_data = utils.group_data(self.dataX, self.dataY)
        grouped_dataY = utils.group_data(self.dataY, self.dataY)

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
