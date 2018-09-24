#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 20:35:05 2018

@author: 2021rahul
"""

import data
import config
import algorithm
import numpy as np
import utils


def LDA1dProjection():
    # PREPARE DATA
    dataset = data.Boston()
    dataset.load_dataset()
    dataset.categorize_data()

    # PROJECT TO 1 DIMENSION
    lda = algorithm.LDA(dimensions=1)
    lda.get_projections(dataset)
    eigen_vectors = lda.eigens
    dataX = utils.project_data(dataset.dataX, eigen_vectors)
    utils.plot_histograms(dataX, dataset.dataY)


def logisticRegression(filename, num_splits , train_percent):
    if(filename=="DIGITS"):
        dataset = data.Digits()
    elif(filename=="BOSTON"):
        dataset = data.Boston()
    dataset.load_dataset()
    dataset.get_kfold_splits(num_splits)
    accuracy = []
    for i in range(num_splits):
        print(i)
        trainX, trainY, testX, testY = dataset.generate_data()
        train_use = int((train_percent*len(trainX))/100)
        LR = algorithm.LogisticRegression([64,10])
        LR.train(trainX[:train_use,:], trainY[:train_use,:], 0.00001, 1000)
        accuracy.append(LR.test(testX, testY))
    print(accuracy)

if __name__ == "__main__":
    logisticRegression("DIGITS", 10, 100)
