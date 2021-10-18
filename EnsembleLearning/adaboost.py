import pandas as pd
import numpy as np
import math
test_df = pd.read_csv("./bank/test.csv", header=None)
train_df = pd.read_csv("./bank/train.csv", header=None)
m_train = len(train_df)
m_test = len(test_df)

########################### preprocess the data 
attrib_name = {0:"age", 1:"job",2:"marital",3:"education",4:"default",5:"balance", 6:"housing",7:"loan",8:"contact",9:"day",10:"month",11:"duration",12:"campaign",13:"pdays",14:"previous",15:"poutcome"}
label_values = ["yes", "no"]

categorical_attrib_values = { "job":     ["admin.","unknown","unemployed","management","housemaid","entrepreneur","student", "blue-collar","self-employed","retired","technician","services"],\
                 "marital":  ["married","divorced","single"],\
                 "education":["unknown","secondary","primary","tertiary"],\
                 "default":  ["yes","no"],\
                 "housing":  ["yes","no"],\
                 "loan":     ["yes","no"],\
                 "contact":  ["unknown","telephone","cellular"],\
                 "month":    ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"],\
                 "poutcome": ["unknown","other","failure","success"]
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

Dtest  = [1/m_test]  * m_test
test_df_processed[17] = Dtest

####################################### calculates the information gain
def entropy_gain(S,Label_col_index, attrib_idx):
    H_S = S.groupby(Label_col_index)[Label_col_index + 1]\
    .apply(lambda x: (x.sum()/S.shape[0])*np.log2(x.sum()/S.shape[0]))\
    .sum()*-1
    
    Expected_H_Sv = S.groupby([attrib_idx,Label_col_index],as_index=False)[Label_col_index+1].sum()\
    .groupby(attrib_idx)[Label_col_index+1].apply(lambda x:(x.sum()/S.shape[0])*((x/x.sum())*np.log2(x/x.sum()))).sum()*-1
        
    return H_S - Expected_H_Sv

####################################### id3 and it prediction fucntions
# returns the column index of the best splitter attribute
# S: set of examples
# Attributes: list of attributes to be evaluated
# splitter_algorithm: the splitter algorithm, can be one of the 3 values ("ME":Majority Error, "GI":Gini Index, "EN":Entropy)
def Best_spliter_attribute(S, Attributes, Label_col_index, splitter_algorithm):
    if len(Attributes) < 2:
        return Attributes[0]
    best_gain = 0
    best_attribute = Attributes[0]
    for v in Attributes:
        if v != Label_col_index:
            gain_v = 0
            if splitter_algorithm == "EN":
                gain_v = entropy_gain(S,Label_col_index, v)
            elif splitter_algorithm == "ME":
                gain_v = ME_gain(S,Label_col_index,v)
            elif splitter_algorithm == "GI":
                gain_v = gini_gain(S,Label_col_index,v)
            else:
                assert False, "Unknown splitter_algorithm:" + splitter_algorithm + "!!!"
            if gain_v > best_gain:
                best_gain = gain_v
                best_attribute = v
#     print("best attrib is:",best_attribute)
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
    for idx, row in S.iterrows():
        all += 1
        gold_label = row[Label_col_index]
        predicted_label = predict(root, row, Label_col_index)
        if predicted_label == gold_label:
            correct +=1
    return correct / all # accuracy 

def get_Et(S, root, Label_col_index):
    Et = 0
    ht_x = []
    for idx, row in S.iterrows():
        gold_label = row[Label_col_index]
        predicted_label = predict(root, row, Label_col_index)
        ht_x.append(predicted_label)
        if predicted_label != gold_label:
            Et += row[Label_col_index + 1]
    return Et, ht_x
        
# ##############              ID3 implementation:
# Input:
# S: the set of Examples
# Attributes: the set of measured attributes
# Label_col_index: column index of the target attribute (the prediction)
# max_tree_level: bounds the height of the tree
# splitter_algorithm: can be one of the 3 values ("ME":Majority Error, "GI":Gini Index, "EN":Entropy)
def ID3(S, Attributes, Label_col_index, max_tree_level, splitter_algorithm):
    if(max_tree_level == 0):                                                            # if at max level
        return S[Label_col_index].mode()[0]   
    elif S[Label_col_index].nunique() == 1:                                             # if all examples have same label:   
        return S[Label_col_index].mode()[0]
    elif len(Attributes) == 0:                                                          # if Attributes empty
        return S[Label_col_index].mode()[0]
    else:
        # 1. Create a Root node for tree
        Root = [] # each "attribute node" is a list s.t. 
                                                    # 1st element = string attribute name
                                                    # 2nd element = dictionary children;
                                                            # key = each possible attribute value v
                                                            # value = an "attribute node" list;  or a string label for leaf nodes
        # 2. A = attribute in Attributes that best splits S
        A = Best_spliter_attribute(S, Attributes, Label_col_index, splitter_algorithm)
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
                Root[1][v] = S[Label_col_index].mode()[0] # string label
            else:
                Attrib_minus_A = Attributes
                if len(Attrib_minus_A) > 0 and A in Attrib_minus_A:
                    Attrib_minus_A.remove(A)
                Root[1][v] = ID3(Sv, Attrib_minus_A,Label_col_index, max_tree_level-1,splitter_algorithm) # an "attribute node" list;
        return Root

# ##############              test the decision stump
print("Some  Training examples ...")
Attributes = [0,1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # initially put all attributes except the label in Attributes set
tree_infoGain = ID3(train_df_processed, Attributes,16,1, "EN")  # note the max_height is set to 2 to construct decision stumps
print(tree_infoGain)

print("#######")
print("train accuracy:",predict_dataset(train_df_processed, tree_infoGain,16))
print("test accuracy:",predict_dataset(test_df_processed, tree_infoGain,16))

# ##############              AdaBoost_train implementation:
# Input:
# S: the set of Examples
# Attributes: the set of measured attributes
# Label_col_index: column index of the target attribute (the prediction)
# T: number of iterations
def AdaBoost_train(S, Attributes, Label_col_index, T):
    print("AdaBoost Training...")
    print("Total iterations:",T)
    
    # 1. Initialize D1(i) = 1/m for all i = 1, 2, ..., m
    h=[]
    alpha=[]
    # 2. For t = 1, 2, ..., T:
    for t in range(T):
        print("iteration:",t)
        # 1. Find a classifier ht whose weighted classification error is better than chance
        Hyp = ID3(S, Attributes, Label_col_index, 1, "EN")
#         print("stump_",t ," test accuracy:",predict_dataset(test_df_processed, Hyp,16))
        h.append(Hyp)
        # compute the Et
        Et, ht_x = get_Et(S,Hyp, Label_col_index)
        # compute Zt
        Zt = S[Label_col_index+1].sum()
        
#         Zt = 2*math.sqrt(Et * (1 - Et))
        # 2. Compute its vote
        Alp = 0.5 * math.log((1-Et)/Et)
        alpha.append(Alp)
        # 3. Update the  values of the weights for the training examples
        temp = S[Label_col_index+1].mul(pd.Series(ht_x))
        temp = temp.apply(lambda x: math.exp(-1*Alp*x))
        S[Label_col_index+1] = (S[Label_col_index+1]/Zt).mul(temp)
    # 3. Return the final hypothesis
    return h, alpha

# ##############              AdaBoost_predict implementation:
# Input: 
# h[1:T]: list of decision stumps
# alpha[1:T]: list of votes
# x: example to be predicted
def AdaBoost_predict(h, alpha, x, Label_col_index):
    ht_x =[]
    for t in range(len(h)):
        ht_x.append(predict(h[t],x,Label_col_index))
    return np.sign(np.dot(ht_x, alpha))

def AdaBoost_predict_dataset(S, h,alpha, Label_col_index):
    all = 0
    correct = 0
    for idx, row in S.iterrows():
        all += 1
        gold_label = row[Label_col_index]
        predicted_label = AdaBoost_predict(h,alpha, row, Label_col_index)
        if predicted_label == gold_label:
            correct +=1
    return correct / all # accuracy 

# ##############              test the AdaBoost
Attributes = [0,1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # initially put all attributes except the label in Attributes set
h, alpha = AdaBoost_train(train_df_processed, Attributes, 16, 20)
print("traininig set prediction...")
print("train accuracy:",AdaBoost_predict_dataset(train_df_processed,h,alpha,16))
print("test set prediction...")
print("test accuracy:",AdaBoost_predict_dataset(test_df_processed,h,alpha,16))