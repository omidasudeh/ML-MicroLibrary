# # Imports and data preprocessing
import pandas as pd
import numpy as np
import math
from scipy.optimize import minimize, rosen, rosen_der

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


# # SVM Implementation
################### standard perceptron methods
def get_error(x,y,w):
    n_rows = x.shape[0]
    predictions = np.sign(np.matmul(x, np.reshape(w, (-1,1))))    # predictions = sign(Wt*x)
    predictions = np.reshape(predictions,(1,-1))
    incorrect_predictions = predictions - y
    count_incorrect_predictions = np.count_nonzero(incorrect_predictions)
    Error = count_incorrect_predictions/ n_rows
    return Error

def Primal_SVM(X,Y,learningRate, alpha, T, C, schedule=0):
    rows = X.shape[0]
    cols = X.shape[1]
    w = np.zeros(cols)                              # 1. Initialize w = 0 ∈ ℜn
    indices = np.arange(rows)
    
    for epoch in range(T):                          # 2. For epoch = 1 … T:
        np.random.shuffle(indices)                       #1. Shuffle the data
        x = X[indices,:]
        y = Y[indices]
        l = learningRate / (1 + learningRate/alpha * epoch)
        if schedule == 1:
            l = learningRate / (1 + epoch)
        for i in range(rows):                            #2. For each training example (xi, yi) ∈ D:
            if np.sum(x[i] * w) * y[i] <= 1:                  #If yi wTxi ≤ 1, w←w−𝛾t.[w0;0]+𝛾t.C.N.yi.xi
                w = w - learningRate * w + learningRate * rows * C * y[i] * x[i]
    return w                                        # 3. Return w


# # Main

print("######################## Primal SVM ########################\n")
print("==== schedule = γt = γ0/(1+(γ0/a)*t) ====")
C = 100/873
w = Primal_SVM(train_x, train_y, 0.5, 0.1, 100, C)
print("C:",C)
print('Primal SVM train Error:', get_error(train_x, train_y,w))
print('Primal SVM test Error: ', get_error(test_x, test_y,w))
print("====")

C = 500/873
w = Primal_SVM(train_x, train_y, 0.5, 0.1, 100, C)
print("C:",C)
print('Primal SVM train Error:', get_error(train_x, train_y,w))
print('Primal SVM test Error: ', get_error(test_x, test_y,w))
print("====")

C = 700/873
w = Primal_SVM(train_x, train_y, 0.5, 0.1, 100, C)
print("C:",C)
print('Primal SVM train Error:', get_error(train_x, train_y,w))
print('Primal SVM test Error: ', get_error(test_x, test_y,w))
print("====")

print("\n==== schedule = γt = γ0/(1+t) ====")
C = 100/873
w = Primal_SVM(train_x, train_y, 0.5, 0.1, 100, C,1)
print("C:",C)
print('Primal SVM train Error:', get_error(train_x, train_y,w))
print('Primal SVM test Error: ', get_error(test_x, test_y,w))
print("====")

C = 500/873
w = Primal_SVM(train_x, train_y, 0.5, 0.1, 100, C,1)
print("C:",C)
print('Primal SVM train Error:', get_error(train_x, train_y,w))
print('Primal SVM test Error: ', get_error(test_x, test_y,w))
print("====")

C = 700/873
w = Primal_SVM(train_x, train_y, 0.5, 0.1, 100, C,1)
print("C:",C)
print('Primal SVM train Error:', get_error(train_x, train_y,w))
print('Primal SVM test Error: ', get_error(test_x, test_y,w))
print("====")

print("######################## Dual SVM ########################\n")

