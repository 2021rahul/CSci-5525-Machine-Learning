#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:20:43 2018

@author: 2021rahul
"""

import numpy as np
import matplotlib.pyplot as plt


def mean(array):
    return array.mean(axis=0)


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


def between_class_scatter(X, Y):
    classes = np.unique(Y[:, 0])
    overall_mean = mean(X)

    grouped_data = group_data(X, Y)
    class_mean = []
    for key in grouped_data:
        class_mean.append(mean(grouped_data[key]))

    Sb = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(classes)):
        val = np.reshape(class_mean[i] - overall_mean, (len(class_mean[i]), 1))
        Sb += np.multiply(len(grouped_data[str(int(i))]), np.dot(val, val.T))
    return Sb


def within_class_scatter(X, Y):
    classes = np.unique(Y[:, 0])

    grouped_data = group_data(X, Y)
    class_mean = []
    for key in grouped_data:
        class_mean.append(mean(grouped_data[key]))

    Sw = np.zeros((X.shape[1], X.shape[1]))
    for i in range(len(classes)):
        val = np.subtract(X.T, np.reshape(class_mean[i], (len(class_mean[i]), 1)))
        Sw = np.add(Sw, np.dot(val, val.T))
    return Sw


def project_data(X, directions):
    return np.dot(X, directions)

colors = ["red", "yellow", "green", "blue"]
def plot_histograms(X, Y):
    grouped_data = group_data(X, Y)
    num_classes = len(np.unique(Y[:, 0]))
    data = [grouped_data[key] for key in grouped_data]
    plt.hist(data, color=colors[:num_classes])
    plt.show()


def sigmoid(X):
    return 1.0/(1.0 + np.exp(-1.0 * X))


def softmax(weights, bias, x):
    output = np.dot(x, weights)
    output = np.add(output, bias)
    output = np.exp(output)
    output = output.T/np.sum(output, axis=1)
    return output.T
