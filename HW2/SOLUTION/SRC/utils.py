#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:20:43 2018

@author: 2021rahul
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import os
import config


def randomize_data(X, Y):
    index = random.sample(range(0, len(Y)), len(Y))
    X = X[index, :]
    Y = Y[index, :]
    return X, Y


def group_data(X, Y):
    classes = np.unique(Y[:, 0])
    grouped_data = {}
    for i in classes:
        indexes = np.where(Y[:, 0] == i)[0]
        grouped_data[str(int(i))] = X[indexes, :]
    return grouped_data


def plot_accuracy(accuracy, stddev, percentage):
    plt.plot(percentage, accuracy, label='Accuracy')
    plt.plot(percentage, stddev, label='Stddev')
    plt.show()
