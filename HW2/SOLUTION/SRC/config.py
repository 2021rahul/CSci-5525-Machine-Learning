#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:20:43 2018

@author: 2021rahul
"""

import os

# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('../')
DATA_DIR = os.path.join(ROOT_DIR, 'DATASET/')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'OUTPUT/')
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

threshold = 1e-5
num_splits = 10
