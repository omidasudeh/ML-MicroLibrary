import pandas as pd
import numpy as np
import math

train_raw = pd.read_csv("./bank-note/train.csv", header=None).values
train_cols = train_raw.shape[1]
train_rows = train_raw.shape[0]
train_x = np.copy(train_raw)
train_x = np.delete(np.concatenate([np.ones((train_rows,1)),train_x], axis=1), -1,1) # augment the bias 1
train_y = train_raw[:, train_cols - 1]
train_y[train_y > 0] = 1      # map 1 -> 1
train_y[train_y == 0] = -1    # map 0 -> -1

test_raw = pd.read_csv("./bank-note/test.csv", header=None).values
test_cols = test_raw.shape[1]
test_rows = test_raw.shape[0]
test_x = np.copy(test_raw)
test_x = np.delete(np.concatenate([np.ones((test_rows,1)),test_x], axis=1), -1,1) # augment the bias 1
test_y = test_raw[:, test_cols - 1]
test_y[test_y > 0] = 1
test_y[test_y == 0] = -1


# # NN Implementation
def pred(x, W):
    predictions = []
    for example in x:
        predictions.append(forward_pass(example, W)[-1])
    return predictions
def get_error(x,y,W):
    n_rows = x.shape[0]
    predictions = np.sign(pred(x, W))    # predictions = sign(Wt*x)
    predictions = np.reshape(predictions,(1,-1))
    incorrect_predictions = predictions - y
    count_incorrect_predictions = np.count_nonzero(incorrect_predictions)
    Error = count_incorrect_predictions/ n_rows
    return Error

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(y, a):
    return 0.5 * (y-a)**2
###### forward pass
## given an augmented example x(n, 1) and the corresponsing weight matrix W(n1, n) 
## returns the next level's activations (n1, 1)
## does not apply sigmoid if is_last_layer = true
def forward_step(x, w, is_last_layer):
    activations = np.reshape(np.dot(np.reshape(x,(1,-1)),w),(-1,1))
    if is_last_layer:
        return activations
    return sigmoid(activations)

## given an augmented example x(n, 1) 
## and the list of weight matrices W[w0, w1,w2] corresponsing to the weights each layer should be multiplied
## returns the final output and activations of the layers
def forward_pass(x, W):
    a = np.reshape(x,(-1,1))
    activations = [a]
    for layer in range(len(W)-1):
        a = forward_step(a, W[layer], False)   # compute the activations of layes[0,n-2]
        a[0] = 1                               # set the bias term
        activations.append(a)
    activations.append(forward_step(a, W[len(W)-1], True))       # compute the last layer output
    return activations

###### backward pass
## given an augmented example x(n, 1)
## given the true label y
## given the list of layers weight matrices W
## given list of layer activations
## returns DW [Dw0, dw1, ...] that is the derivatives of L with respect to weights of each layer
def backward_pass(x,y, W,A):
    DA = A[len(A)-1] - y # derivatives L with respect to the activations of this layer, initially DA  = dL/dy = y - y*
    DW = []              # list of derivatives with repect to the weights of all layers
    for layer in reversed(range(len(W))): #0,1,2
        a = A[layer]               # activations of this layer
        a_next = A[layer+1]        # activations of next layer 
        t = 0
        if layer == (len(W) - 1):  # if the last layer no sigmoid derivation
            t = np.reshape(DA,(-1,1))
        else:
            t = np.reshape(a_next * (1 - a_next) * DA,(-1,1))

        DW.insert(0,np.matmul(a,np.reshape(t, (1,-1))))
        if layer != 0:
            DA = np.matmul(W[layer], t)
            DA[0] = 0                 # drop the derivation with respect to bias
    return DW
        
##### Stocastic Gradient Decent
def SGD(X,Y, W,learningRate, alpha, T):
    rows = X.shape[0]
    cols = X.shape[1]
    indices = np.arange(rows)
    for epoch in range(T):                          # 2. For epoch = 1 … T:
        np.random.shuffle(indices)                       #1. Shuffle the data
        x = X[indices,:]
        y = Y[indices]
        r = learningRate / (1 + learningRate/alpha * epoch)
        for i in range(rows):                            #2. For each training example (xi, yi) ∈ D:
            A  = forward_pass(x[i], W)                       # compute Activations
            DW = backward_pass(x[i],y[i], W, A)              # compute Weight Gradients
            for i in range (len(W)):
                W[i] = W[i] - r*DW[i]                        # update W <-- W - rDW
        
    return W                                        # 3. Return w


# # Main

print("###########################")
print("question 2.a")
print("This is to demonstrate forward and backward are working:")
##### W just for debug
w1 = np.array(             [              [0,-1,1],              [0,-2,2],              [0,-3,3]             ]            )
w2 = np.array(             [              [0,-1,1],              [0,-2,2],              [0,-3,3]             ]             )
w3 = np.array(             [              [-1],              [2],              [-1.5]             ]             )
W = [w1,w2,w3]
for w in W:
    print (w)
    print("=====")

###########################
## testing the Forward and Backward passes
#   run the >> Forward Pass << and get the activations for each layer 
x0 = [1,1,1]
y0 = 1
print("Forward-Pass Activations:")
A = forward_pass(x0, W)
print(A)
print("=====")

#   run the >> Backward Pass << and get the derivations for each layer ####
print("Backward-Pass Gradients of Weight matrices:")
DW = backward_pass(x0,y0, W, A)
print(DW)
print("=====")
###########################

###########################
print("###########################")
print("question 2.b")
for nodes in [5,10,25,50,100]:
    layerOneSize = nodes
    layerTwoSize = nodes
    LayerSizes = [train_cols, layerOneSize, layerTwoSize, 1]

    # randomly initializes the weight matrix of each layer
    W = [np.random.randn(LayerSizes[layer], LayerSizes[layer+1]) for layer in range(len(LayerSizes)-1)]
    for i in range(len(W)-1):               # set the bias weight columns to 0 (except for output layer)
        W[i][:,0 ] = 0
    W = SGD(train_x, train_y, W, 0.05,0.1, 30)
    print("For ",nodes,"nodes the Train Error is:",get_error(train_x, train_y, W))
    print("For ",nodes,"nodes the Test Error is:",get_error(test_x, test_y, W))
    print("=====")
###########################

###########################
print("###########################")
print("question 2.c")
for nodes in [5,10,25,50,100]:
    layerOneSize = nodes
    layerTwoSize = nodes
    LayerSizes = [train_cols, layerOneSize, layerTwoSize, 1]

    # randomly initializes the weight matrix of each layer
    W = [np.zeros((LayerSizes[layer], LayerSizes[layer+1])) for layer in range(len(LayerSizes)-1)]
    for i in range(len(W)-1):               # set the bias weight columns to 0 (except for output layer)
        W[i][:,0 ] = 0
    W = SGD(train_x, train_y, W, 0.05,0.1, 30)
    print("For ",nodes,"nodes the Train Error is:",get_error(train_x, train_y, W))
    print("For ",nodes,"nodes the Test Error is:",get_error(test_x, test_y, W))
    print("=====")
###########################

