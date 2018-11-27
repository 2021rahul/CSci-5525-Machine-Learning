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
import timeit


def myDualSVM(filename, C):
    # PREPARE DATA
    dataset = data.DATA()
    dataset.read_data(filename)

    # RUN SVM ON THE SPLITS
    train_accuracies = np.zeros((config.num_splits, len(C)))
    test_accuracies = np.zeros((config.num_splits, len(C)))
    num_sv = np.zeros((config.num_splits, len(C)))
    weights_norm = np.zeros((config.num_splits, len(C)))
    for i in range(config.num_splits):
        trainX, trainY, testX, testY = dataset.generate_data()
        for j, c in enumerate(C):
            SVM = algorithm.SVM(c)
            SVM.train(trainX, trainY)
            weights_norm[i, j] = np.linalg.norm(SVM.weights)
            num_sv[i, j] = SVM.num_sv
            train_accuracies[i, j] = SVM.test(trainX, trainY)
            test_accuracies[i, j] = SVM.test(testX, testY)

    # MEAN ERROR AND STANDARD DEVIATION
    return train_accuracies, test_accuracies, num_sv, weights_norm


def myPegasos(filename, k, numruns):
    # PREPARE DATA
    dataset = data.DATA()
    dataset.read_data(filename)

    # RUN SVM ON THE SPLITS
    time = []
    loss_values = []
    iter_values = []
    weights_norm = []
    for i in range(numruns):
        print(i)
        dataX = dataset.dataX
        dataY = dataset.dataY
        SVM = algorithm.SVM_Pegasos(dataX.shape[1])
        start = timeit.default_timer()
        iter_vals, loss_vals = SVM.train(dataX, dataY, k)
        
        loss_values.append(loss_vals)
        iter_values.append(iter_vals)
        stop = timeit.default_timer()
        time.append(stop-start)

    # MEAN ERROR AND STANDARD DEVIATION
    mean = np.mean(time)
    stddev = np.std(time)
    print("Avg Time: ", mean)
    print("Standard Deviation in Time: ", stddev)
    return mean, stddev, loss_values, iter_values


def mySoftplus(filename, k, numruns):
    # PREPARE DATA
    dataset = data.DATA()
    dataset.read_data(filename)

    # RUN SVM ON THE SPLITS
    time = []
    loss_values = []
    iter_values = []
    for i in range(numruns):
        print(i)
        dataX = dataset.dataX
        dataY = dataset.dataY
        SVM = algorithm.SVM_Softplus(dataX.shape[1])
        start = timeit.default_timer()
        iter_vals, loss_vals = SVM.train(dataX, dataY, k)
        loss_values.append(loss_vals)
        iter_values.append(iter_vals)
        stop = timeit.default_timer()
        time.append(stop-start)

    # MEAN ERROR AND STANDARD DEVIATION
    mean = np.mean(time)
    stddev = np.std(time)
    print("Avg Time: ", mean)
    print("Standard Deviation in Time: ", stddev)
    return mean, stddev, loss_values, iter_values


if __name__ == "__main__":

#     SVM CLASSIFIER ON MNIST-13 DATASET
    C = [0.01, 0.1, 1, 10, 100]
    print("SVM CLASSIFIER ON MNIST-13 DATASET")
    train_accuracies, test_accuracies, num_sv, weights_norm = myDualSVM("MNIST-13.csv", C)
    mean_train_acc = np.mean(train_accuracies, axis=0)
    std_train_acc = np.std(train_accuracies, axis=0)
    mean_test_acc = np.mean(test_accuracies, axis=0)
    std_test_acc = np.std(test_accuracies, axis=0)
    mean_support_vectors = np.mean(num_sv, axis=0).astype(int)
    std_support_vectors = np.std(num_sv, axis=0)
    average_norm_weight = np.mean(weights_norm, axis=0)
    for j, c in enumerate(C):
        print("C: ", c)
        print("Average Number of Support Vectors: ", mean_support_vectors[j], " Standard Deviation: ", std_support_vectors[j])
        print("Average Test Accuracy: ", mean_test_acc[j], " Standard Deviation: ", std_test_acc[j])
        print("Average Margin", 1/average_norm_weight[j])
    utils.plot_accuracy(mean_train_acc, std_train_acc, C, "SVM_train")
    utils.plot_accuracy(mean_test_acc, std_test_acc, C, "SVM_test")
    utils.plot_accuracy(mean_support_vectors, std_support_vectors, C, "SVM_num_sv")
    print("\n######################################################\n")

#     SVM PEGASOS ON MNIST-13 DATASET
    print("SVM PEGASOS ON MNIST-13 DATASET")
    K = [1, 20, 200, 1000, 2000]
    avg_runtime = []
    stddev_runtime = []
    for k in K:
        print("k: ", k)
        mean, stddev, loss_values, iter_values = myPegasos("MNIST-13.csv", k, 5)
        utils.plot_loss(loss_values, iter_values, "pegasos"+str(k))
        avg_runtime.append(mean)
        stddev_runtime.append(stddev)
    utils.plot_accuracy(avg_runtime, stddev_runtime, K, "runtime_pegasos")
    print("\n######################################################\n")

    # SVM SOFTPLUS ON MNIST-13 DATASET
    print("SVM SOFTPLUS ON MNIST-13 DATASET")
    K = [1, 20, 200, 1000, 2000]
    avg_runtime = []
    stddev_runtime = []
    for k in K:
        print("k: ", k)
        mean, stddev, loss_values, iter_values = mySoftplus("MNIST-13.csv", k, 5)
        utils.plot_loss(loss_values, iter_values, "softplus"+str(k))
        avg_runtime.append(mean)
        stddev_runtime.append(stddev)
    utils.plot_accuracy(avg_runtime, stddev_runtime, K, "runtime_softplus")
    print("\n######################################################\n")
