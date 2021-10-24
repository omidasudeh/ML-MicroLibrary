#!/usr/bin/env python
# coding: utf-8

# # Read the input Data

import pandas as pd
import numpy as np
import math
import copy as cp
import sys
import pickle
train_raw = pd.read_csv("./concrete/train.csv", header=None).values
cols = train_raw.shape[1]
rows = train_raw.shape[0]
train_x = np.copy(train_raw)
train_x[:,cols - 1] = 1
train_y = train_raw[:, cols - 1]

test_raw = pd.read_csv("./concrete/test.csv", header=None).values
cols = test_raw.shape[1]
rows = test_raw.shape[0]
test_x = np.copy(test_raw)
test_x[:,cols - 1] = 1
test_y = test_raw[:, cols - 1]


print(np.reshape(train_y, (-1,1)).shape)


# # Batched LMS Implementation


def Gradien_Decent(x,y):
    m = x.shape[0]
    n = x.shape[1]
    Error = 1
    threshold = 10e-6
    Max_iter = 10000
    LearningRate = 0.01
    w = np.zeros([n,1])
    Costs = []
    iter = 0
    while Error > threshold and iter < Max_iter:
        iter = iter + 1
        diff = np.matmul(x, w) - np.reshape(y, (-1,1))      
        gradient = np.sum(x * (diff), axis = 0)
        w1 = w - LearningRate*np.reshape(gradient,(-1,1))
        Error = np.sum(np.abs(w - w1))
        w = w1
        cost = 0.5 * np.sum(np.square(diff))
        Costs.append([iter, cost])
    print("converged after ", iter, "iterations.")
    return w, Costs



w, Costs = Gradien_Decent(train_x, train_y)
print("Batched Gradien_Decent W vector:")
print(w)
from numpy import savetxt
savetxt('Costs.csv', Costs, delimiter=',')


Test_Cost = 0.5 * np.sum(np.square(np.reshape(np.squeeze(np.matmul(test_x,w)) - test_y, (-1,1))))
print('Batched Test_Cost:', Test_Cost)


# # Stocastic LMS Implementation


def Stoc_Gradien_Decent(x,y):
    m = x.shape[0]
    n = x.shape[1]
    Error = 1
    threshold = 10e-6
    Max_iter = 100000
    LearningRate = 0.01
    w = np.zeros([n,1])
    Costs = []
    iter = 0
    while Error > threshold and iter < Max_iter:
        iter = iter + 1
        idx = np.random.randint(m,size=1)
        x1 = x[idx]
        y1 = y[idx]
        SG = (y1 - np.dot(x1, w))*x1 
        w1 = w + LearningRate* np.reshape(SG, (-1,1))
        Error = np.sum(np.abs(w - w1))
        w = w1
        cost = 0.5 * np.sum(np.square(np.matmul(x, w) - np.reshape(y, (-1,1))))
        Costs.append([iter, cost])
    print("converged after ", iter, "iterations.")
    return w, Costs



w, Costs = Stoc_Gradien_Decent(train_x, train_y)
print("Stocastic Gradien_Decent W vector:")
print(w)
from numpy import savetxt
savetxt('Stoc_Costs.csv', Costs, delimiter=',')




Test_Cost = 0.5 * np.sum(np.square(np.reshape(np.squeeze(np.matmul(test_x,w)) - test_y, (-1,1))))
print('Stocastic Test_Cost:', Test_Cost)


# # Analytical LMS Implementation


def Analytical_Gradien_Decent(x, y):
    x_t = np.transpose(x)
    return np.matmul(np.linalg.inv(np.matmul(x_t, x)),np.matmul(x_t, y))



w = Analytical_Gradien_Decent(train_x, train_y)
Test_Cost = 0.5 * np.sum(np.square(np.reshape(np.squeeze(np.matmul(test_x,w)) - test_y, (-1,1))))
print("Analytical Gradien_Decent W vector:")
print(w)
print('Analytical Test_Cost:', Test_Cost)