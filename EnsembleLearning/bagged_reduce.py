# python3 bagged_reduce.py bagged_trees/ > bagged_clean.csv
import pandas as pd
import numpy as np
import math
import copy as cp
import sys
import pickle
test_df = pd.read_csv("./bank/test.csv", header=None)
m_test = len(test_df)

train_df = pd.read_csv("./bank/train.csv", header=None)
m_train = len(train_df)

######## convert label: yes->+1 no-> -1
train_df[16] = train_df[16].apply(lambda x: -1 if x=='no' else 1)
test_df[16]  = test_df[16].apply(lambda x: -1 if x=='no' else 1)


print("iterations, train_accuracy, test_accuracy")

# # read all the trees predictions into a list
Trees = []
for root, dirs, files in os.walk(sys.argv[1], topdown=False):
	if files:	
		for filename in files:
			if ".out" not in filename: 
				continue
			file_path = os.path.join(root, filename)
			with open(file_path, 'rb') as handle:
				package = pickle.load(handle)
				Trees.append(package)
m = len(Trees[0][2])
# # take majority for predictions of all trees
train_prediction_frequency = [[0,0]]*m
test_prediction_frequency  = [[0,0]]*m

for T in Trees:
	train_accuracy,test_accuracy, hx_train,hx_test = T
	for i in range (m):
		if int(hx_train[i]) == 1:
			train_prediction_frequency[i][0] += 1
		elif int(hx_train[i]) == -1:
			train_prediction_frequency[i][1] += 1
		
		if int(hx_test[i]) == 1:
			test_prediction_frequency[i][0] += 1
		elif int(hx_test[i]) == -1:
			test_prediction_frequency[i][1] += 1

final_hx_train  = []
final_hx_test   = []
for i in range (m):
	train_label = 0
	if(train_prediction_frequency[i][0] > train_prediction_frequency[i][1]):
		train_label = 1
	else:
		train_label = -1
	final_hx_train.append[train_label]

	test_label = 0
	if(test_prediction_frequency[i][0] > test_prediction_frequency[i][1]):
		test_label = 1
	else:
		test_label = -1
	final_hx_test.append[test_label]


# # compute the train and test accuracies for the final hx train and test
train_correct_predictions = (pd.Series(final_hx_train) - train_df[16]).append(lambda x: abs(x)).sum()
print("train_accuracy", train_correct_predictions/m_train)

test_correct_predictions = (pd.Series(final_hx_test) - test_df[16]).append(lambda x: abs(x)).sum()
print("test_accuracy", test_correct_predictions/m_test)