#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 09:20:43 2018

@author: 2021rahul
"""


import numpy as np
import random
import cvxopt
import config


class SVM():

    def __init__(self, C):
        self.weights = None
        self.bias = 0
        self.num_sv = None
        self.C = C

    def predict(self, x):
        output = np.dot(x, self.weights)
        output = np.add(output, self.bias)
        return np.sign(output)

    def form_qp_problem(self, dataX, dataY):
        K = np.dot(dataX, dataX.T)
        P = cvxopt.matrix(np.dot(dataY, dataY.T) * K)
        q = cvxopt.matrix(np.ones(len(dataX)) * -1)
        A = cvxopt.matrix(dataY.T)
        b = cvxopt.matrix(0.0)
        G = cvxopt.matrix(np.concatenate((np.diag(np.ones(len(dataX))*-1), np.identity(len(dataX)))))
        h = cvxopt.matrix(np.concatenate((np.zeros(len(dataX)), np.ones(len(dataX)) * self.C)))
        return K, P, q, G, h, A, b

    def train(self, dataX, dataY):
        K, P, q, G, h, A, b = self.form_qp_problem(dataX, dataY)
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        a = np.ravel(solution['x'])
        sv_index = np.where(a > config.threshold)[0]
        self.num_sv = len(sv_index)
        a = np.reshape(a[sv_index], (-1, 1))
        X = dataX[sv_index]
        Y = dataY[sv_index]
        self.weights = np.reshape(np.sum(a*Y*X, axis=0), (-1, 1))
        for n in range(len(a)):
            self.bias += Y[n]
            self.bias -= np.sum(a * Y * K[sv_index[n], sv_index])
            self.bias /= len(a)

    def test(self, dataX, dataY):
        prediction = self.predict(dataX)
        count = np.sum(prediction == dataY)
        return count/len(dataY)

class SVM_Pegasos():

    def __init__(self, shape):
        self.weights = np.zeros((shape, 1))
        self.regularizer = 0.0002
        self.max_iter = 100*2000

    def forward(self, x):
        return np.dot(x, self.weights)

    def predict(self, x):
        output = self.forward(x)
        output[output > 0] = 1
        output[output < 0] = -1
        return output

    def objective(self, x, y):
        loss = 1-y*self.forward(x)
        loss[loss < 0] = 0
#        print(np.sum(loss)/len(y), self.regularizer*np.power(np.linalg.norm(self.weights), 2))
        loss_obj = np.sum(loss)/len(y) + self.regularizer*np.power(np.linalg.norm(self.weights), 2)
        return loss_obj

    def train(self, dataX, dataY, k):
        iter = 0
        relative_change = 1
        loss_vals = []
        iter_vals = []
        while iter < self.max_iter and relative_change>1e-5:
            objective_loss = self.objective(dataX, dataY)
            index = random.sample(range(0, len(dataX)), k)
            trainX = dataX[index]
            trainY = dataY[index]
            evaluate = trainY*self.forward(trainX)
            new_index = np.where(evaluate<1)[0]
            iter += k
            step_size = 1/(self.regularizer*iter)
            self.weights = (self.weights*(1-(self.regularizer*step_size))) + np.reshape((step_size/k)*np.sum(trainY[new_index]*trainX[new_index], axis=0), (-1, 1))
            self.weights = min(1, (1/np.sqrt(self.regularizer))/np.linalg.norm(self.weights)) * self.weights
            new_objective_loss = self.objective(dataX, dataY)
            relative_change = abs((new_objective_loss-objective_loss)/new_objective_loss)
            iter_vals.append(iter)
            loss_vals.append(new_objective_loss)
        return iter_vals, loss_vals

class oldSVM_Pegasos():

    def __init__(self, shape):
        self.weights = np.zeros((shape, 1))
        self.regularizer = 0.1
        self.max_iter = 100*2000

    def forward(self, x):
        return np.dot(x, self.weights)

    def predict(self, x):
        output = self.forward(x)
        output[output > 0] = 1
        output[output < 0] = -1
        return output

    def objective(self, x, y):
        loss = 1-y*self.forward(x)
        loss[loss < 0] = 0
        loss_obj = np.sum(loss)/len(y) + self.regularizer*np.power(np.linalg.norm(self.weights), 2)
        return loss_obj

    def train(self, dataX, dataY, k):
        iter = 0
        relative_change = 1
        loss_vals = []
        iter_vals = []
        while iter < self.max_iter and relative_change>1e-7:
            objective_loss = self.objective(dataX, dataY)
            index = random.sample(range(0, len(dataX)), k)
            trainX = dataX[index]
            trainY = dataY[index]
            evaluate = trainY*self.forward(trainX)
            new_index = np.where(evaluate<1)[0]
            iter += 1
            step_size = 1/(self.regularizer*iter)
            self.weights = (self.weights*(1-(self.regularizer*step_size))) + np.reshape((step_size/k)*np.sum(trainY[new_index]*trainX[new_index], axis=0), (-1, 1))
            new_objective_loss = self.objective(dataX, dataY)
            relative_change = abs((new_objective_loss-objective_loss)/new_objective_loss)
            iter_vals.append(iter)
            loss_vals.append(new_objective_loss)
        return iter_vals, loss_vals

class SVM_Softplus():

    def __init__(self, shape):
        self.weights = np.random.uniform(low=0, high=0.002, size=(shape, 1))
        self.regularizer = 0.0001
        self.max_iter = 100*2000
        self.a = 100

    def forward(self, x):
        return np.dot(x, self.weights)

    def predict(self, x):
        output = self.forward(x)
        output[output > 0] = 1
        output[output < 0] = -1
        return output

    def objective(self, x, y):
        loss = np.sum(np.log(1 + np.exp((1-y*self.forward(x))/self.a)))
        loss_obj = loss*self.a/len(y) + self.regularizer*np.power(np.linalg.norm(self.weights), 2)
        return loss_obj

    def gradient(self, x, y):
        exp_val = (1 - y*self.forward(x))/self.a
        exp_v = 1/(1+np.exp(-exp_val))
        gradient = x.T*y*exp_v
        return gradient

    def train(self, dataX, dataY, k):
        iter = 0
        relative_change = 1
        loss_vals = []
        iter_vals = []
        while iter < self.max_iter and relative_change > 1e-7:
            objective_loss = self.objective(dataX, dataY)
            index = random.sample(range(0, len(dataX)), k)
            iter += 1
            step_size = 1/(self.regularizer*iter)
            grad = 0
            for i in index:
                x = dataX[i]
                y = dataY[i]
                grad += self.gradient(x, y)
            self.weights = ((1-2*self.regularizer*step_size)*self.weights) + np.reshape((step_size/k)*grad, (-1, 1))
            new_objective_loss = self.objective(dataX, dataY)
            relative_change = abs((new_objective_loss-objective_loss)/new_objective_loss)
            iter_vals.append(iter)
            loss_vals.append(new_objective_loss)
        return iter_vals, loss_vals

class oldSVM_Softplus():

    def __init__(self, shape):
        self.weights = np.zeros((shape, 1))
        self.regularizer = 0.1
        self.max_iter = 100*2000
        self.a = 100

    def forward(self, x):
        return np.dot(x, self.weights)

    def predict(self, x):
        output = self.forward(x)
        output[output > 0] = 1
        output[output < 0] = -1
        return output

    def objective(self, x, y):
        loss = np.sum(np.log(1 + np.exp((1-y*self.forward(x))/self.a)))
        loss_obj = loss*self.a/len(y) + self.regularizer*np.power(np.linalg.norm(self.weights), 2)
        return loss_obj

    def gradient(self, x, y):
        exp_val = (1 - y*self.forward(x))/self.a
        exp_v = 1/(1+np.exp(-exp_val))
        gradient = x.T*y*exp_v
        return gradient

    def train(self, dataX, dataY, k):
        iter = 0
        relative_change = 1
        loss_vals = []
        iter_vals = []
        while iter < self.max_iter and relative_change > 1e-7:
            objective_loss = self.objective(dataX, dataY)
            index = random.sample(range(0, len(dataX)), k)
            iter += 1
            step_size = 1/(self.regularizer*iter)
            grad = 0
            for i in index:
                x = dataX[i]
                y = dataY[i]
                grad += self.gradient(x, y)
            self.weights = ((1-2*self.regularizer*step_size)*self.weights) + np.reshape((step_size/k)*grad, (-1, 1))
            new_objective_loss = self.objective(dataX, dataY)
            relative_change = abs((new_objective_loss-objective_loss)/new_objective_loss)
            iter_vals.append(iter)
            loss_vals.append(new_objective_loss)
        return iter_vals, loss_vals