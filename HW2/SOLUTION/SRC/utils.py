#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:20:43 2018

@author: 2021rahul
"""

import numpy as np
import random
import matplotlib.pyplot as plt
import config
import os


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


def plot_accuracy(accuracy, stddev, X, name):
    plt.plot(np.log10(X), accuracy, label='Accuracy')
    plt.plot(np.log10(X), stddev, label='Stddev')
    plt.savefig(os.path.join(config.OUTPUT_DIR, name+".png"))
    plt.close()


def plot_loss(loss_values, iter_values, name):
    for i in range(5):
        plt.plot(np.log10(iter_values[i]), loss_values[i])
    plt.savefig(os.path.join(config.OUTPUT_DIR, name+".png"))
    plt.close()
