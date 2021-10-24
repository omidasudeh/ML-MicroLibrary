#!/usr/bin/env python
# coding: utf-8

# # read the data


import pandas as pd
import numpy as np
import math
import copy as cp
import sys
import pickle
import random

test_df = pd.read_csv("../data/bank/test.csv", header=None)
m_test = len(test_df)

train_df = pd.read_csv("../data/bank/train.csv", header=None)
m_train = len(train_df)

# train_df = pd.read_csv("./bank/sample1.csv", header=None)
# m_train = len(train_df)


# # preprocess the data



attrib_name = {0:"age", 1:"job",2:"marital",3:"education",4:"default",5:"balance", 6:"housing",7:"loan",8:"contact",9:"day",10:"month",11:"duration",12:"campaign",13:"pdays",14:"previous",15:"poutcome"}
label_values = [1, -1]

categorical_attrib_values = { "job":     ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"],                 "marital":  ["married","divorced","single"],                 "education":["unknown","secondary","primary","tertiary"],                 "default":  ["yes","no"],                 "housing":  ["yes","no"],                 "loan":     ["yes","no"],                 "contact":  ["unknown","telephone","cellular"],                 "month":    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],                 "poutcome": ["unknown","other","failure","success"]
                }
def convert_numerical_to_binary(df, Label_col_index):
    for col_idx in range(Label_col_index):
        if attrib_name[col_idx] not in categorical_attrib_values:
            df[col_idx] = pd.to_numeric(df[col_idx], errors='coerce')
            threashold = df[col_idx].median()
            print("preprocessing numerical attribute:", attrib_name[col_idx], "threashold:", threashold)
            df[col_idx] = df[col_idx].apply(lambda x: 0 if int(x) <= threashold else 1)

######## convert the numerical to binary
train_df_processed = train_df
test_df_processed = test_df
convert_numerical_to_binary(train_df_processed, 16)
convert_numerical_to_binary(test_df_processed, 16)

######## convert label: yes->+1 no-> -1
train_df_processed[16] = train_df_processed[16].apply(lambda x: -1 if x=='no' else 1)
test_df_processed[16] = test_df_processed[16].apply(lambda x: -1 if x=='no' else 1)

######## the the initial weights as a column to the dataframe
Dtrain = [1/m_train] * m_train
train_df_processed[17] = Dtrain
# print(train_df_processed)
Dtest  = [1/m_test]  * m_test
test_df_processed[17] = Dtest


# # gain functions


# calculates the information gain
def entropy_gain(S,Label_col_index, attrib_idx): 
    H_S = S.groupby(Label_col_index)[Label_col_index + 1]    .apply(lambda x: (x.sum())*np.log2(x.sum()))    .sum()*-1
    
#     val0 = S.groupby([attrib_idx,Label_col_index])[Label_col_index].count()
    val0 = S.groupby([attrib_idx,Label_col_index])[Label_col_index+1].sum()
    val1 = S.groupby([attrib_idx])[Label_col_index+1].sum()
#     print(val0)
#     print(val1)
    val2 = val0/val1
    val3 = val2.apply(lambda x: x*math.log2(x)).reset_index(name='plog2p')
    val4 = val3.groupby([attrib_idx])["plog2p"].sum()*-1
    val5 = S.groupby([attrib_idx])[attrib_idx].apply(lambda x: x.count()/S.shape[0])

    Expected_H_Sv = (val4*val5).sum()

#     print("H_S:",H_S, "Expected_H_Sv",Expected_H_Sv, "gain:",H_S - Expected_H_Sv)
    return H_S - Expected_H_Sv


# # id3 and prediction fucntions

# returns the label that has the max weight
# S: set of examples
def select_label_with_max_weight_sum(S, Label_col_index):
    return S.groupby([Label_col_index])[Label_col_index+1].sum().idxmax()
    
# returns the column index of the best splitter attribute
# S: set of examples
# Attributes: list of attributes to be evaluated
# splitter_algorithm: the splitter algorithm, can be one of the 3 values ("ME":Majority Error, "GI":Gini Index, "EN":Entropy)
def Best_spliter_attribute(S, Attributes, Label_col_index, splitter_algorithm):
    if len(Attributes) < 2:
        return Attributes[0]
    best_gain = 0
    best_attribute = Attributes[0] #
    for v in Attributes:
        if v != Label_col_index:
            gain_v = 0
            if splitter_algorithm == "EN":
                gain_v = entropy_gain(S,Label_col_index, v)
#                 print("attrib:",attrib_name[v], "gain:",gain_v)
            else:
                assert False, "Unknown splitter_algorithm:" + splitter_algorithm + "!!!"
            if gain_v > best_gain:
                best_gain = gain_v
                best_attribute = v
#     print("best is:",best_attribute, "GAIN:",best_gain)
    return best_attribute

def numeric_attrib_value(S, attrib_col_idx, numeric_value):
    threashold = S[attrib_col_idx].median()
    return numeric_value >= threashold   

def predict(root, entry, Label_col_index):
    example = {} 
    for i in range(Label_col_index):
        example[attrib_name[i]] = entry[i]
    return predict_helper(root, example)

def predict_helper(root, example):
    if isinstance(root, list): # if attrib-node
        root_attrib_name = root[0]
    else:
        return root
    example_attrib_val = example[root_attrib_name]
    if isinstance(root[1][example_attrib_val], list): # if attrib-node
        return predict_helper(root[1][example_attrib_val], example)
    else: # if leaf node
        return root[1][example_attrib_val]
    
def predict_dataset(S, root, Label_col_index):
    all = 0
    correct = 0
    hx = []
    for idx, row in S.iterrows():
        all += 1
        gold_label = row[Label_col_index]
        predicted_label = predict(root, row, Label_col_index)
        hx.append(predicted_label)
        if predicted_label == gold_label:
            correct +=1
    return correct / all , hx # accuracy, set of predictions
        
# ##############              ID3 implementation:
# Input:
# S: the set of Examples
# Attributes: the set of measured attributes
# Label_col_index: column index of the target attribute (the prediction)
# max_tree_level: bounds the height of the tree
# splitter_algorithm: can be one of the 3 values ("ME":Majority Error, "GI":Gini Index, "EN":Entropy)
def ID3(S, Attributes, Label_col_index, max_tree_level, splitter_algorithm, K):
    if(max_tree_level == 0):                                                            # if at max level
#         print("max_tree_level reached")
        return select_label_with_max_weight_sum(S, Label_col_index)
    elif S[Label_col_index].nunique() == 1:                                             # if all examples have same label:   
#         print("Label_col_index unique")
        return select_label_with_max_weight_sum(S, Label_col_index)
    elif len(Attributes) == 0:                                                          # if Attributes empty
#         print("Attributes")
        return select_label_with_max_weight_sum(S, Label_col_index)
    else:
        # 1. Create a Root node for tree
        Root = [] # each "attribute node" is a list s.t. 
                                                    # 1st element = string attribute name
                                                    # 2nd element = dictionary children;
                                                            # key = each possible attribute value v
                                                            # value = an "attribute node" list;  or a string label for leaf nodes
        # 2. A = attribute in Attributes that best splits S
         # # randomly select k attributes
        G = [] # holds the position of selected attributes
        for i in range(0,K):
            n = random.randint(0,len(Attributes)-1)
            G.append(Attributes[n]) 
        A = Best_spliter_attribute(S, G, Label_col_index, splitter_algorithm)
#         print("best is:",attrib_name[A])
        Root.append(attrib_name[A]) # 1st element = string attribute name
        Root.append({})             # 2nd element = dictionary children;
        # 3. for each possible value v of that A can take:
        attribute_values=[]
        if(attrib_name[A] in categorical_attrib_values):
            attribute_values = categorical_attrib_values[attrib_name[A]]
        else: # o.w. it is numerical 
            attribute_values = [0,1]
        for v in attribute_values:
            # 1. Add a new tree branch corresponding to A=v
            # 2. Let Sv be the subset of examples in S with A=v
            Sv = S.loc[S[A] == v]
            if len(Sv) == 0: # if Sv is empty
#                 print("Sv is empty")
                Root[1][v] = select_label_with_max_weight_sum(S, Label_col_index) # string label
            else:
                Attrib_minus_A = Attributes
                if len(Attrib_minus_A) > 0 and A in Attrib_minus_A:
                    Attrib_minus_A.remove(A)
                Root[1][v] = ID3(Sv, Attrib_minus_A,Label_col_index, max_tree_level-1,splitter_algorithm, K) # an "attribute node" list;
        return Root


# # Random Forest


def RandTreeLearn(S, Attributes, k): 
    tree = ID3(S, Attributes,16,16, "EN", k)
    return tree
    
iters = int(sys.argv[1])
Attributes = [0,1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
Trees = []
k = int(sys.argv[2])
print("Random Forest Training...")
print("k=",k)
print("total iteration:", iters)
for it in range(iters):
    print("iteration:", it)
    # # generate a bootstrap tree 
    train_df_sample = train_df_processed.sample(m_train, replace=True)
    tree = RandTreeLearn(train_df_sample, Attributes, k)
    Trees.append(tree)

print("prediction...")
for number_of_trees in range(1, 1+ len(Trees)):
    # # take majority for predictions of all trees
    train_prediction_frequency = pd.Series([0]*m_train)
    test_prediction_frequency  = pd.Series([0]*m_test)
    for t in range ( number_of_trees):
        T = Trees[t]
        train_accuracy, hx_train = predict_dataset(train_df_processed, T, 16)
        test_accuracy, hx_test = predict_dataset(test_df_processed, T, 16)
        train_prediction_frequency = train_prediction_frequency + pd.Series(hx_train)
        test_prediction_frequency = test_prediction_frequency + pd.Series(hx_test)

    final_hx_train  = train_prediction_frequency.apply(lambda x: np.sign(x))
    final_hx_test  = test_prediction_frequency.apply(lambda x: np.sign(x))

    # # compute the train and test accuracies for the final hx train and test
    train_false_predictions = np.count_nonzero(final_hx_train - train_df[16])
    test_false_predictions = np.count_nonzero(final_hx_test - test_df[16])
    print("number_of_trees:",number_of_trees," ,train_Error_Rate:",train_false_predictions/m_train,",test_Error_Rate", test_false_predictions/m_test)






