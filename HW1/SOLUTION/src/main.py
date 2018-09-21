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


if __name__ == "__main__":

    LDA1dProjection()



