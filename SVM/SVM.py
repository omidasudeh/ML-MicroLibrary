# # Imports and data preprocessing
import pandas as pd
import numpy as np
import math
import scipy.optimize as opt

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
################### Primal SVM
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

################### Dual SVM
def Dual_SVM(x, y, C):
    #### minimize the SVM dual objective function and get the a1*, a2*,..., an* 
    op = opt.minimize(lambda a: 0.5 * np.sum(\
                                             np.matmul(\
                                                       np.multiply(\
                                                                   np.multiply(\
                                                                               np.reshape(a,(-1,1))\
                                                                               ,np.reshape(y, (-1,1))\
                                                                              )\
                                                                   ,x),\
                                                       np.transpose(\
                                                                    np.multiply(\
                                                                                np.multiply(\
                                                                                            np.reshape(a,(-1,1)),\
                                                                                            np.reshape(y, (-1,1))\
                                                                                           )\
                                                                                ,x)\
                                                                   )\
                                                      )\
                                            )- np.sum(a),
                       np.zeros(x.shape[0]),
                       method='SLSQP',\
                       bounds=[(0, C)] * x.shape[0],\
                       constraints={'type': 'eq',\
                                     'fun': lambda a: np.matmul(\
                                                                np.reshape(a,(1, -1)),\
                                                                np.reshape(y,(-1,1)))[0]\
                                    },\
                       options={'disp': False}\
                    )
    #### Recover W and b
    w = np.sum(np.multiply(np.multiply(np.reshape(op.x,(-1,1)), np.reshape(y, (-1,1))), x), axis=0).tolist()
    b = np.mean(y[np.where((op.x > 0) & (op.x < C))] - np.matmul(x[np.where((op.x > 0) & (op.x < C)),:], np.reshape(w, (-1,1))))
    return np.reshape(np.array(w + [b]), (5,1))

################### Gaussian SVM
def get_gaus_error(x0, y0, x, y, alpha, gamma):
    predictions = np.sign(np.reshape(np.sum(np.multiply(np.reshape(alpha, (-1,1)), np.multiply(np.reshape(y0, (-1,1)), Kernel(x0, x, gamma))), axis=0), (-1,1)))
    incorrect_predictions = predictions - np.reshape(y,(-1,1))
    count_incorrect_predictions = np.count_nonzero(incorrect_predictions)
    return count_incorrect_predictions/ y.shape[0]

def Kernel(x1, x2, gamma):
    ## k(xi, xj) = exp(-(||xi-xj||2)/γ)
    return np.reshape(\
                        np.exp(\
                                -1 * np.sum(\
                                            np.square(\
                                                        np.reshape(np.tile(x1, (1, x2.shape[0])),\
                                                        (-1,x1.shape[1])) - np.tile(x2, (x1.shape[0], 1))\
                                                     ),\
                                            axis=1) / gamma\
                                ),\
                        (x1.shape[0], x2.shape[0])\
                    )

def Gaus_SVM(x, y, C, gamma):
    k = Kernel(x,x,gamma)
    return opt.minimize(\
                        lambda alpha: 0.5 * np.sum(\
                                                    np.multiply(\
                                                                np.matmul(\
                                                                            np.multiply(\
                                                                                        np.reshape(alpha,(-1,1)),\
                                                                                        np.reshape(y, (-1,1))\
                                                                                        ),\
                                                                            np.transpose(\
                                                                                        np.multiply(\
                                                                                                    np.reshape(alpha,(-1,1)),\
                                                                                                    np.reshape(y, (-1,1))\
                                                                                                    )\
                                                                                        )\
                                                                        ),\
                                                                k)\
                                                    )- np.sum(alpha),\
                        np.zeros(x.shape[0]),\
                        method='SLSQP',\
                        bounds=[(0, C)] * x.shape[0],\
                        constraints={'type': 'eq',\
                                     'fun': lambda alpha: np.matmul(\
                                                                    np.reshape(alpha,(1, -1)),\
                                                                    np.reshape(y,(-1,1))\
                                                                    )[0]\
                                    },\
                        options={'disp': False}\
                    ).x

# # Main

print("############################  Primal SVM")
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

print("############################  Dual SVM")

C = 100/873
print("C:",C)
trainX = train_x[:,[x for x in range(train_cols - 1)]]
testX  = test_x [:,[x for x in range(test_cols  - 1)]]
w = Dual_SVM( trainX ,train_y, C)
print('Dual SVM train Error:', get_error(train_x, train_y,w))
print('Dual SVM test Error:', get_error(test_x, test_y,w))
print("====")

C = 500/873
print("C:",C)
w = Dual_SVM( trainX ,train_y, C)
print('Dual SVM train Error:', get_error(train_x, train_y,w))
print('Dual SVM test Error:', get_error(test_x, test_y,w))
print("====")

C = 700/873
print("C:",C)
w = Dual_SVM( trainX ,train_y, C)
print('Dual SVM train Error:', get_error(train_x, train_y,w))
print('Dual SVM test Error:', get_error(test_x, test_y,w))
print("====")

print("############################ Gaussian SVM")
for c in [100/873, 500/873, 700/873]:
    for gamma in [.1,.5,1,5,100]:
        alpha = Gaus_SVM(trainX ,train_y, c, gamma)
        print("Setting: c=", c, "γ=", gamma)
        print('Gaussian SVM train Error:', get_gaus_error(trainX, train_y, trainX, train_y, alpha, gamma))
        print('Gaussian SVM test Error:', get_gaus_error(trainX, train_y, testX, test_y,alpha, gamma))
        print("**********")