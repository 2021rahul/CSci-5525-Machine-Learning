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
    # PREPARE DATA
    if(filename=="DIGITS"):
        dataset = data.Digits()
    elif(filename=="BOSTON"):
        dataset = data.Boston()
    dataset.load_dataset()
    dataset.get_kfold_splits(num_splits)

    # RUN LOGISTIC REGRESSION ON THE SPLITS
    accuracy = []
    for i in range(num_splits):
#        print(i)
        trainX, trainY, testX, testY = dataset.generate_data()
        train_use = int((train_percent*len(trainX))/100)
        LR = algorithm.LogisticRegression([trainX.shape[1], trainY.shape[1]])
        LR.train(trainX[:train_use,:], trainY[:train_use, :], 0.00001, 1000)
        accuracy.append(LR.test(testX, testY))
    return accuracy

def naiveBayesGaussian(filename, num_splits , train_percent):
    # PREPARE DATA
    if(filename=="DIGITS"):
        dataset = data.Digits()
    elif(filename=="BOSTON"):
        dataset = data.Boston()
    dataset.load_dataset()
    dataset.get_kfold_splits(num_splits)

    # RUN NAIVE BAYES GAUSSIAN ON THE SPLITS
    accuracy = []
    for i in range(num_splits):
#        print(i)
        trainX, trainY, testX, testY = dataset.generate_data()
        trainY = trainY.argmax(axis=1)
        testY = testY.argmax(axis=1)
        train_use = int((train_percent*len(trainX))/100)
        NB = algorithm.NaiveBayes()
        NB.train(trainX[:train_use,:], trainY[:train_use])
        accuracy.append(NB.test(testX, testY))
    return accuracy


if __name__ == "__main__":

    LRaccuracy10 = np.asarray(logisticRegression("DIGITS", 10, 10))/180
    LRaccuracy25 = np.asarray(logisticRegression("DIGITS", 10, 25))/180
    LRaccuracy50 = np.asarray(logisticRegression("DIGITS", 10, 50))/180
    LRaccuracy75 = np.asarray(logisticRegression("DIGITS", 10, 75))/180
    LRaccuracy100 = np.asarray(logisticRegression("DIGITS", 10, 100))/180

    NBaccuracy10 = np.asarray(naiveBayesGaussian("DIGITS", 10, 10))/180
    NBaccuracy25 = np.asarray(naiveBayesGaussian("DIGITS", 10, 25))/180
    NBaccuracy50 = np.asarray(naiveBayesGaussian("DIGITS", 10, 50))/180
    NBaccuracy75 = np.asarray(naiveBayesGaussian("DIGITS", 10, 75))/180
    NBaccuracy100 = np.asarray(naiveBayesGaussian("DIGITS", 10, 100))/180
