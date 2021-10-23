# python3 bagged_reduce.py bagged_trees_predictions/ > bagged_clean.csv
import pandas as pd
import numpy as np
import math
import copy as cp
import sys, os
import pickle
test_df = pd.read_csv("./bank/test.csv", header=None)
m_test = len(test_df)

train_df = pd.read_csv("./bank/train.csv", header=None)
m_train = len(train_df)

######## convert label: yes->+1 no-> -1
train_df[16] = train_df[16].apply(lambda x: -1 if x=='no' else 1)
test_df[16]  = test_df[16].apply(lambda x: -1 if x=='no' else 1)


print("iterations, train_error, test_error")

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

for number_of_trees in range(1, 1+ len(Trees)):
	# # take majority for predictions of all trees
	train_prediction_frequency = pd.Series([0]*m_train)
	test_prediction_frequency  = pd.Series([0]*m_test)
	for t in range ( number_of_trees):
		T = Trees[t]
		train_accuracy,test_accuracy, hx_train,hx_test = T
		train_prediction_frequency = train_prediction_frequency + pd.Series(hx_train)
		test_prediction_frequency = test_prediction_frequency + pd.Series(hx_test)

	final_hx_train  = train_prediction_frequency.apply(lambda x: np.sign(x))
	final_hx_test  = test_prediction_frequency.apply(lambda x: np.sign(x))

	# # compute the train and test accuracies for the final hx train and test
	train_false_predictions = np.count_nonzero(final_hx_train - train_df[16])
	test_false_predictions = np.count_nonzero(final_hx_test - test_df[16])

	print(number_of_trees,",",train_false_predictions/m_train,",", test_false_predictions/m_test)