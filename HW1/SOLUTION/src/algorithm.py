#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:20:43 2018

@author: 2021rahul
"""


import utils
import numpy as np


class LDA():

    def __init__(self, dimensions):
        self.dimensions = dimensions
        self.eigens = None

    def get_projections(self, data):
        Sb = utils.between_class_scatter(data.dataX, data.dataY)
        Sw = utils.within_class_scatter(data.dataX, data.dataY)
        mat = np.dot(np.linalg.inv(Sw), Sb)
        self.eigens = utils.get_sorted_eigens(mat)
        self.eigens = np.reshape(self.eigens[self.dimensions, 1:], (len(self.eigens), self.dimensions))
