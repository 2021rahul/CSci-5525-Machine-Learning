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
    projection_vectors = lda.w
    dataX = utils.project_data(dataset.dataX, projection_vectors)
    utils.plot_histograms(dataX, dataset.dataY)


def LDA2dGaussGM(num_crossval):
    # PREPARE DATA
    dataset = data.Digits()
    dataset.load_dataset()

    # PROJECT TO 2 DIMENSION
    lda = algorithm.LDA(dimensions=2)
    lda.get_projections(dataset)
    projection_vectors = lda.w
    dataset.dataX = utils.project_data(dataset.dataX, projection_vectors)

    # RUN GAUSSIAN GENERATIVE MODELLING ON THE SPLITS
    accuracy = []
    for i in range(num_crossval):
        trainX, trainY, testX, testY = dataset.generate_data()
        GM = algorithm.Gaussian_Generative()
        GM.train(trainX, trainY)
        accuracy.append(GM.test(testX, testY)/len(testY))

    mean = np.mean(accuracy)
    stddev = np.std(accuracy)

    print("Mean Accuracy: ", mean)
    print("Standard Deviation in Accuracy: ", stddev)
    return mean, stddev


def logisticRegression(filename, num_splits , train_percent):
    # PREPARE DATA
    if(filename=="DIGITS"):
        dataset = data.Digits()
        dataset.load_dataset()
        dataset.one_hot_encoded()
    elif(filename=="BOSTON"):
        dataset = data.Boston()
        dataset.load_dataset()
        dataset.categorize_data()

    # RUN LOGISTIC REGRESSION ON THE SPLITS
    accuracy = []
    for i in range(num_splits):
        trainX, trainY, testX, testY = dataset.generate_data()
        train_use = int((train_percent*len(trainX))/100)
        if(filename=="DIGITS"):
            LR = algorithm.LogisticRegression([trainX.shape[1], trainY.shape[1]])
            LR.train(trainX[:train_use,:], trainY[:train_use, :], 0.00001, 1000)
        else:
            LR = algorithm.LogisticRegression_2class([trainX.shape[1], trainY.shape[1]])
            LR.train(trainX[:train_use,:], trainY[:train_use, :], 0.0000001, 4000)
        accuracy.append(LR.test(testX, testY)/len(testY))

    mean = np.mean(accuracy)
    stddev = np.std(accuracy)

    print("Mean Accuracy: ", mean)
    print("Standard Deviation in Accuracy: ", stddev)
    return mean, stddev


def naiveBayesGaussian(filename, num_splits , train_percent):
    # PREPARE DATA
    if(filename=="DIGITS"):
        dataset = data.Digits()
        dataset.load_dataset()
    elif(filename=="BOSTON"):
        dataset = data.Boston()
        dataset.load_dataset()
        dataset.categorize_data()
    dataset.one_hot_encoded()

    # RUN NAIVE BAYES GAUSSIAN ON THE SPLITS
    accuracy = []
    for i in range(num_splits):
        trainX, trainY, testX, testY = dataset.generate_data()
        trainY = trainY.argmax(axis=1)
        testY = testY.argmax(axis=1)
        train_use = int((train_percent*len(trainX))/100)
        NB = algorithm.NaiveBayes()
        NB.train(trainX[:train_use,:], trainY[:train_use])
        accuracy.append(NB.test(testX, testY)/len(testY))

    mean = np.mean(accuracy)
    stddev = np.std(accuracy)

    print("Mean Accuracy: ", mean)
    print("Standard Deviation in Accuracy: ", stddev)
    return mean, stddev


if __name__ == "__main__":

    # PROJECT BOSTON DATA TO 1 DIMENSION
    print("PROJECT BOSTON DATA TO 1 DIMENSION")
    LDA1dProjection()
    print("\n######################################################\n")

     # CLASSIFY DIGITS DATA USING GAUSSIAN GENERATIVE MODELLING
    print("GAUSSIAN GENERATIVE MODELLING ON DIGITS DATASET PROJECTED TO 2 DIMENSIONS")
    LDA2dGaussGM(num_crossval=10)
    print("\n######################################################\n")

    # LOGISTIC REGRESSION CLASSIFIER ON DIGITS DATASET
    print("LOGISTIC REGRESSION CLASSIFIER ON DIGITS DATASET")
    mean_accuracy = []
    mean_stddev = []
    for train_percent in config.percentage:
        mean, stddev = logisticRegression("DIGITS", 10, train_percent)
        mean_accuracy.append(mean)
        mean_stddev.append(stddev)
    utils.plot_accuracy(mean_accuracy, mean_stddev, config.percentage)
    print("\n######################################################\n")

    # LOGISTIC REGRESSION CLASSIFIER ON BOSTON DATASET
    print("LOGISTIC REGRESSION CLASSIFIER ON BOSTON DATASET")
    mean_accuracy = []
    mean_stddev = []
    for train_percent in config.percentage:
        mean, stddev = logisticRegression("BOSTON", 10, train_percent)
        mean_accuracy.append(mean)
        mean_stddev.append(stddev)
    utils.plot_accuracy(mean_accuracy, mean_stddev, config.percentage)
    print("\n######################################################\n")

    # NAIVE BAYES CLASSIFIER ON DIGITS DATASET
    print("NAIVE BAYES CLASSIFIER ON DIGITS DATASET")
    mean_accuracy = []
    mean_stddev = []
    for train_percent in config.percentage:
        mean, stddev = naiveBayesGaussian("DIGITS", 10, train_percent)
        mean_accuracy.append(mean)
        mean_stddev.append(stddev)
    utils.plot_accuracy(mean_accuracy, mean_stddev, config.percentage)
    print("\n######################################################\n")

    # NAIVE BAYES CLASSIFIER ON BOSTON DATASET
    print("NAIVE BAYES CLASSIFIER ON BOSTON DATASET")
    mean_accuracy = []
    mean_stddev = []
    for train_percent in config.percentage:
        mean, stddev = naiveBayesGaussian("BOSTON", 10, train_percent)
        mean_accuracy.append(mean)
        mean_stddev.append(stddev)
    utils.plot_accuracy(mean_accuracy, mean_stddev, config.percentage)
