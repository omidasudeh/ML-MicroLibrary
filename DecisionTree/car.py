import pandas as pd
import numpy as np
import math

test_df = pd.read_csv("./car/test.csv", header=None)
train_df = pd.read_csv("./car/train.csv", header=None)

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

attrib_name = {0:"buying", 1:"maint",2:"doors",3:"persons",4:"lug_boot",5:"safety"}
label_values = ["unacc", "acc", "good", "vgood"]

attrib_values = { "buying":   ["vhigh", "high", "med", "low"],\
                 "maint":    ["vhigh", "high", "med", "low"],\
                 "doors":    ["2", "3", "4", "5more"],\
                 "persons":  ["2", "4", "more"],\
                 "lug_boot": ["small", "med", "big"],\
                 "safety":   ["low", "med", "high"]
                }

def predict(root, entry):
    example = {} 
    for i in range(6):
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
        predicted_label = predict(root, row)
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
    if(max_tree_level ==0):                                                             # if at max level
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
        Root.append({})            # 2nd element = dictionary children;
        # 3. for each possible value v of that A can take:
        for v in attrib_values[attrib_name[A]]:
            # 1. Add a new tree branch corresponding to A=v
            # 2. Let Sv be the subset of examples in S with A=v
            Sv = S.loc[S[A] == v]
            if len(Sv) == 0:
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
Attributes = [0,1,2,3,4,5] # initially put all attributes except the label in Attributes set
tree_gini = ID3(train_df, Attributes,6, 6, "GI")
print(tree_gini)

print("\nIG-tree")
print("#######")
Attributes = [0,1,2,3,4,5] # initially put all attributes except the label in Attributes set
tree_entopy = ID3(train_df, Attributes,6, 10, "EN")
print(tree_entopy)

print("\nME-tree")
print("#######")
Attributes = [0,1,2,3,4,5] # initially put all attributes except the label in Attributes set
tree_ME = ID3(train_df, Attributes,6, 10, "ME")
print(tree_ME)

print("\n\n########## experiment #############")
print("this may take few minutes dependeng to you machine (for me it took less than 5 mins to complete)")
for hight in range(1,7):
    print("tree hight:", hight)
    for app in ["EN", "GI", "ME"]:
        print("gain approach:", app)
        print("Training ...")
        Attributes = [0,1,2,3,4,5] # initially put all attributes except the label in Attributes set
        tree = ID3(train_df, Attributes,6, hight, app)
        
        print("prediction ...")
        print("train accuracy:",predict_dataset(train_df, tree,6))
        print("test accuracy: ", predict_dataset(test_df, tree,6))
        print("###############")