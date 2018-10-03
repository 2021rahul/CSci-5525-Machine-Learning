#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:20:43 2018

@author: 2021rahul
"""

import numpy as np
import random
import matplotlib.pyplot as plt


def mean(array):
    return np.mean(array, axis=0)


def stddev(array):
    return np.std(array, axis=0)


def randomize_data(X, Y):
    index = random.sample(range(0, len(Y)), len(Y))
    X = X[index, :]
    Y = Y[index, :]
    return X, Y


def get_sorted_eigens(array):
    eigen_value, eigen_vector = np.linalg.eig(array)
    eigens = np.concatenate((np.reshape(eigen_value, (len(eigen_value), 1)), eigen_vector), axis=1)
    return eigens[np.argsort(eigens[:, 0])]


def group_data(X, Y):
    classes = np.unique(Y[:, 0])
    grouped_data = {}
    for i in classes:
        indexes = np.where(Y[:, 0] == i)[0]
        grouped_data[str(int(i))] = X[indexes, :]
    return grouped_data


def project_data(X, directions):
    return np.dot(X, directions)


def plot_histograms(X, Y):
    colors = ["green", "red", "yellow", "blue"]
    grouped_data = group_data(X, Y)
    num_classes = len(np.unique(Y[:, 0]))
    data = [grouped_data[key] for key in grouped_data]
    plt.hist(data, color=colors[:num_classes], bins=40)
    plt.show()


def sigmoid(X):
    return 1.0/(1.0 + np.exp(-1.0 * X))


def softmax(weights, bias, x):
    output = np.dot(x, weights)
    output = np.add(output, bias)
    output = np.exp(output)
    output = output.T/np.sum(output, axis=1)
    return output.T
