import pandas as pd
import numpy as np
import math
test_df = pd.read_csv("./bank/test.csv", header=None)
train_df = pd.read_csv("./bank/train.csv", header=None)

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
            df[col_idx] = df[col_idx].apply(lambda x: "0" if int(x) < threashold else "1") 

######## preprocess the data
print("########### preprocessing the data ##########")
train_df_processed = train_df
test_df_processed = test_df
convert_numerical_to_binary(train_df_processed, 16)
convert_numerical_to_binary(test_df_processed, 16)

# calculates the information gain
def entropy_gain(S,Label_col_index, attrib_idx):
    H_S = S.groupby(Label_col_index)[Label_col_index]\
    .apply(lambda x: (x.count()/S.shape[0])*np.log2(x.count()/S.shape[0]))\
    .sum()*-1
    
    Expected_H_Sv = S.groupby([attrib_idx,Label_col_index],as_index=False)[Label_col_index].count()\
    .groupby(attrib_idx).apply(lambda x:(x.sum()/S.shape[0])*((x/x.sum())*np.log2(x/x.sum()))).sum()*-1
    return H_S - Expected_H_Sv[Label_col_index]


# calculates the gini gain
def gini_gain(S,Label_col_index, attrib_idx):
    G_S = 1 - (S.groupby(Label_col_index)[Label_col_index]\
    .apply(lambda x: (x.count()/S.shape[0])**2)\
    .sum())
    
    Expected_G_Sv = 1 - (S.groupby([attrib_idx,Label_col_index],as_index=False)[Label_col_index].count()\
    .groupby(attrib_idx).apply(lambda x:(x.sum()/S.shape[0])*(x/x.sum())**2).sum())
    return G_S - Expected_G_Sv[Label_col_index]
    
# calculates the majority error gain
def ME_gain(S,Label_col_index, attrib_idx):  
    freq = S.groupby(Label_col_index)[Label_col_index].count()
    
    ME_S = (freq.sum()- freq.max())/ S.shape[0]

    Expected_ME_Sv = S.groupby([attrib_idx,Label_col_index],as_index=False)[Label_col_index].count()\
    .groupby(attrib_idx).apply(lambda x: (x.sum()/S.shape[0])*(1 - (x.max()/x.sum())))\
    .sum()
    
    return max(0,(ME_S - Expected_ME_Sv[Label_col_index]))

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
    print("best attrib is:",best_attribute)
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
    root_attrib_name = root[0]
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
        
# ##############              ID3 implementation:
# Input:
# S: the set of Examples
# Attributes: the set of measured attributes
# Label_col_index: column index of the target attribute (the prediction)
# max_tree_level: bounds the height of the tree
# splitter_algorithm: can be one of the 3 values ("ME":Majority Error, "GI":Gini Index, "EN":Entropy)
def ID3(S, Attributes, Label_col_index, max_tree_level, splitter_algorithm):
    if(max_tree_level == 0):                                                             # if at max level
        return S[Label_col_index].mode()[0]   
    if S[Label_col_index].nunique() == 1:                                               # if all examples have same label:   
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
            attribute_values = ["0","1"]
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

# ##############              main
print("\n\n######### Some Training/prediction examples ########")
print("gini-tree")
print("#######")
Attributes = [0,1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # initially put all attributes except the label in Attributes set
tree_gini = ID3(train_df_processed, Attributes,16, 16, "GI")
print(tree_gini)
print("train accuracy:",predict_dataset(train_df_processed, tree_gini,16))


print("\n\n########## experiment #############")
print("this may take few minutes dependeng to you machine (for me it took less than 5 mins to complete)")
for hight in range(1,17):
    print("tree hight:", hight)
    for app in ["EN", "GI", "ME"]:
        print("gain approach:", app)
        print("Training ...")
        Attributes = [0,1,2,3,4,5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15] # initially put all attributes except the label in Attributes set
        tree = ID3(train_df_processed, Attributes,16, hight, app)
        
        print("prediction ...")
        print("train accuracy:",predict_dataset(train_df_processed, tree,16))
        print("test accuracy: ", predict_dataset(test_df_processed, tree,16))
        print("###############")

