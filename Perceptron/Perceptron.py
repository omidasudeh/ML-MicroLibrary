# # Imports and data preprocessing
import pandas as pd
import numpy as np
import math

train_raw = pd.read_csv("./bank-note/train.csv", header=None).values
train_cols = train_raw.shape[1]
train_rows = train_raw.shape[0]
train_x = np.copy(train_raw)
train_x[:,train_cols - 1] = 1
train_y = train_raw[:, train_cols - 1]
train_y[train_y > 0] = 1
train_y[train_y == 0] = -1

test_raw = pd.read_csv("./bank-note/test.csv", header=None).values
test_cols = test_raw.shape[1]
test_rows = test_raw.shape[0]
test_x = np.copy(test_raw)
test_x[:,test_cols - 1] = 1
test_y = test_raw[:, test_cols - 1]
test_y[test_y > 0] = 1
test_y[test_y == 0] = -1


# # Perceptron Implementation
################### standard perceptron methods
def get_error(x,y,w):
    n_rows = x.shape[0]
    predictions = np.sign(np.matmul(x, np.reshape(w, (-1,1))))    # predictions = sign(Wt*x)
    predictions = np.reshape(predictions,(1,-1))
    incorrect_predictions = predictions - y
    count_incorrect_predictions = np.count_nonzero(incorrect_predictions)
    Error = count_incorrect_predictions/ n_rows
    return Error

def Standard_Perceptron(X,Y,learningRate, T):
    rows = X.shape[0]
    cols = X.shape[1]
    w = np.zeros(cols)                              # 1. Initialize w = 0 ∈ ℜn
    indices = np.arange(rows)
    
    for epoch in range(T):                          # 2. For epoch = 1 … T:
        np.random.shuffle(indices)                       #1. Shuffle the data
        x = X[indices,:]
        y = Y[indices]
        for i in range(rows):                            #2. For each training example (xi, yi) ∈ D:
            if np.sum(x[i] * w) * y[i] <= 0:                  #If yi wTxi ≤ 0, update w ← w + r yi xi
                w = w + learningRate * y[i] * x[i]
    return w                                        # 3. Return w

################### voted perceptron methods
def get_Voted_error(x, y,W,C):
    n_rows = x.shape[0]
    W = np.transpose(W)
    predictions = np.sign(np.matmul(np.sign(np.matmul(x, W)), C))
    predictions = np.reshape(predictions,(1,-1))
    incorrect_predictions = predictions - y
    count_incorrect_predictions = np.count_nonzero(incorrect_predictions)
    Error = count_incorrect_predictions/ n_rows
    return Error

def Voted_Perceptron(X,Y,learningRate, T):
    rows = X.shape[0]
    cols = X.shape[1]
    w = np.zeros(cols)                              # 1. Initialize w = 0 ∈ ℜn
    m = np.zeros(cols)                              # 1. Initialize m = 0 ∈ ℜn
    indices = np.arange(rows)
    C = np.array([])                                # c0,c1,..., ck
    W = np.array([])                                # w0,w1,..., wk
    c = 0                                           # init c0 = 0
    for epoch in range(T):                          # 2. For epoch = 1 … T:
        np.random.shuffle(indices)                       #1. Shuffle the data
        x = X[indices,:]
        y = Y[indices]
        for i in range(rows):                            #2. For each training example (xi, yi) ∈ D:
            if np.sum(x[i] * w) * y[i] <= 0:                  # If yi wTxi ≤ 0
                W = np.append(W, w)                             # save wm
                C = np.append(C, c)                             # save cm
                w = w + learningRate * y[i] * x[i]              # update wm+1 ← wm + r yi xi
                c = 1                                           # cm = 1
            else:                                             # else
                c += 1                                          # cm = cm+1
    W = np.reshape(W, (C.shape[0],-1))
    C = np.reshape(C, (-1,1))
    return W,C                                        # 3. Return (w1, c1), (w2, c2), …, (wk, Ck)

################### average perceptron methods
def Avg_Perceptron(X,Y,learningRate, T):
    rows = X.shape[0]
    cols = X.shape[1]
    w = np.zeros(cols)                              # 1. Initialize w = 0 ∈ ℜn
    indices = np.arange(rows)
    a = np.zeros(cols)
    for epoch in range(T):                          # 2. For epoch = 1 … T:
        np.random.shuffle(indices)                       #1. Shuffle the data
        x = X[indices,:]
        y = Y[indices]
        for i in range(rows):                            #2. For each training example (xi, yi) ∈ D:
            if np.sum(x[i] * w) * y[i] <= 0:                  #If yi wTxi ≤ 0, update w ← w + r yi xi
                w = w + learningRate * y[i] * x[i]
            a = a + w
    return a                                        # 3. Return w


# # Main

# ## Standard Perceptron
########### Standard Perceptron ##############
w = Standard_Perceptron(train_x, train_y, 0.5, 10)
print("Standard Perceptron\n===========\nw:\n",w)
print('Standard Perceptron Test Error: ', get_error(test_x, test_y,w))


# ## Voted Perceptron
########### Voted Perceptron ##############
W,C = Voted_Perceptron(train_x, train_y, 0.5, 10)
np.savetxt("2b_.csv", W, delimiter=",")
print("\nVoted Perceptron\n===========")
print("W length:",len(W))
print("W:\n",W)
print("C length:",len(C))
print("C:\n",np.reshape(C, (1,-1))[0])
print('Voted Perceptron Test Error: ', get_Voted_error(test_x, test_y,W,C))


# ## Average Perceptron
########### Average Perceptron ##############
w = Avg_Perceptron(train_x, train_y, 0.5, 10)
print("\nAverage Perceptron\n===========\nw:\n",w)
print('Average Perceptron Test Error: ', get_error(test_x, test_y,w))

