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


def myDualSVM(filename, C):
    # PREPARE DATA
    data = data.DATA()
    data.read_data(filename)

    # RUN SVM ON THE SPLITS
    error = []
    for i in range(num_splits):
        trainX, trainY, testX, testY = data.generate_data()
        SVM = algorithm.SVM([trainX.shape[1], trainY.shape[1]], C)
        SVM.train(trainX, trainY)
        prediction = SVM.predict(testX)
        error.append(np.sum(prediction != testY)/len(testY))

    # MEAN ERROR AND STANDARD DEVIATION
    mean = np.mean(error)
    stddev = np.std(error)
    print("Mean Error: ", mean)
    print("Standard Deviation in Error: ", stddev)
    return mean, stddev


if __name__ == "__main__":

    # SVM CLASSIFIER ON MNIST-13 DATASET
    C = [0.01, 0.1, 1, 10, 100]
    mean_error = []
    mean_stddev = []
    print("SVM CLASSIFIER ON MNIST-13 DATASET")
    for c in C:
        print("C: ", c)
        mean, stddev = myDualSVM("MNIST-13.csv", c)
        mean_accuracy.append(mean)
        mean_stddev.append(stddev)
    utils.plot_accuracy(mean_accuracy, mean_stddev, C)
    print("\n######################################################\n")
