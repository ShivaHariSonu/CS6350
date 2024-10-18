#In[1]:
import pickle
import numpy as np
import math
import pandas as pd
import matplotlib.pyplot as plt
import random

columns = ["age","job","marital","education","default","balance","housing","loan","contact","day","month","duration","campaign","pdays","previous","poutcome","y"]

train_data = pd.read_csv("Bank/train.csv",header=None)
test_data = pd.read_csv("Bank/test.csv",header=None)

train_data.columns = columns
test_data.columns = columns

train_data_backup = train_data.copy()

########Train Data Preprocessing###################

def num_to_bin(col,x):
    if x> np.median(train_data[col]):
        return "yes"
    return "no"
for col in ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']:
    train_data[col] = train_data[col].map(lambda x: num_to_bin(col,x))
###################################################

##########Test Data Preprocessing################

def num_to_bin(col,x):
    if x> np.median(train_data_backup[col]):
        return "yes"
    return "no"
for col in ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']:
    test_data[col] = test_data[col].map(lambda x: num_to_bin(col,x))
###################################################
    
X_train = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]

X_test = test_data.iloc[:,:-1]
Y_test = test_data.iloc[:,-1]


class ID3:
    def __init__(self):
        self.data = None
        self.features = None
        self.labels = None
        
    def total_entropy(self,data):
        label_data = data['y'].value_counts()
        entropy = 0
        for i in label_data:
            entropy -= (i/sum(label_data)) * math.log2(i/sum(label_data))
        return entropy
    
    def fea_cat_entropy(self,data, feature):
        categories_features = data[feature].value_counts().keys()
        cat_entropy = 0
        total_data_length = len(data)
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['y'].value_counts()
            s = 0
            for i in label_fea_data:
                s -= (i/sum(label_fea_data)) * math.log2(i/sum(label_fea_data))
            cat_entropy += (sum(label_fea_data)/total_data_length) * (s)
        return cat_entropy
    
    def total_gini(self,data):
        label_data = data['y'].value_counts()
        entropy = 1
        for i in label_data:
            entropy -= (i/sum(label_data))**2
        return entropy
    
    def fea_cat_gini(self,data, feature):
        categories_features = data[feature].value_counts().keys()
        cat_entropy = 0
        total_data_length = len(data)
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['y'].value_counts()
            s = 1
            for i in label_fea_data:
                s -= (i/sum(label_fea_data))**2
            cat_entropy += (sum(label_fea_data)/total_data_length) * (s)
        return cat_entropy
    
    def total_me(self,data):
        label_data = data['y'].value_counts()
        return (min(label_data)/sum(label_data)) if len(label_data)>1 else 0
    
    def fea_cat_me(self,data, feature):
        categories_features = data[feature].value_counts().keys()
        cat_entropy = 0
        total_data_length = len(data)
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['y'].value_counts()
            if len(label_fea_data)!=4:
                   #As the min is always 0 for those
                   cat_entropy+=0 
            else:
                cat_entropy += (min(label_fea_data)/total_data_length)
        return cat_entropy
    
    def IG(self,data,features,split_method):
        if split_method=="entropy":
          return self.total_entropy(data) - self.fea_cat_entropy(data,features)
        elif split_method=="gini":
          return self.total_gini(data) - self.fea_cat_gini(data,features)
        elif split_method=="majorityerror":
          return self.total_me(data) - self.fea_cat_me(data,features)
        

    def create_root_node(self,data,features,split_method):
        total_fea_ig = dict()
        for i in features:
            total_fea_ig[i] = self.IG(data,i,split_method)
        best_feature = max(total_fea_ig, key=total_fea_ig.get)
        return best_feature
    
    def find_bestsplits(self,data,data_copy,root_node):
        features_groups = dict()
        temp_node_values = list(data[root_node].value_counts().keys())
        root_node_values = list(data_copy[root_node].value_counts().keys())
        for x in temp_node_values:
            for i in range(len(data)):
                if data.iloc[i][root_node] == x:
                    if features_groups.get(x,0)==0:
                        features_groups[x] = [(dict(data.iloc[i][:-1]),data.iloc[i]['y'])]
                    else:
                        features_groups[x].append((dict(data.iloc[i][:-1]),data.iloc[i]['y']))
        for x in root_node_values:  
            if x not in features_groups:  
                label_counts = dict(data['y'].value_counts())
                features_groups[x] =  [({}, max(label_counts,key=label_counts.get))]
        return features_groups
        
    def ID3_Algo(self,data,data_copy,features,split_method,backup_features=[]):
        if len(data['y'].value_counts()) == 1:
            return data['y'].value_counts().keys()
        if len(features)==0: 
            label_counts = dict(data['y'].value_counts())
            return max(label_counts,key=label_counts.get)
        
        #Finding the node that has best split
        root_node = self.create_root_node(data,features,split_method)
        if(root_node in features):
            features.remove(root_node)
            backup_features.append(root_node)
            
        #Splitting based on that node
        feature_groups = self.find_bestsplits(data, data_copy,root_node)
        subtree_dict = {}
        for i ,j in feature_groups.items():
            data = pd.DataFrame.from_dict({k: dict(v) for k,v in pd.DataFrame(j)[0].items()}, orient='index')
            data['y'] = pd.DataFrame(j)[1]
            subtree_dict[i] = self.ID3_Algo(data, data_copy, features,split_method,backup_features)
        final_tree = (root_node,subtree_dict)
        last_feature = backup_features.pop()
        features.append(last_feature)
        return final_tree

def classify(tree, query):
      if tree in labels:
          return tree
      key = query.get(tree[0])
      if key not in tree[1]:
          key = None
      class_ = classify(tree[1][key], query)
      return class_
  
def groupclassify(all_classifiers,sample):
    pred = []
    for classifier in all_classifiers:
        if classify(classifier,sample)=="yes":
            pred.append("yes")
        else:
            pred.append("no")
    return max(set(pred), key = pred.count)

def random_split_data(train_data,samplesize,fea_size):
    sa = random.choices(range(train_data.shape[0]),k=samplesize)
    row_samples = train_data.iloc[sa].reset_index(drop=True)
    samples = random.sample(range(len(train_data.columns)-1),k=fea_size)
    #Adding the label column
    samples.append(16)
    sampled_data = row_samples.iloc[:,samples]
    return sampled_data

all_trees = dict()
single_tree_pred = dict()
multi_tree_predictions = dict()
split_method = "entropy"
labels = ['yes','no']

for i in range(100):
    single_tree_pred_for_one_tree = []
    multi_tree_pred = []
    all_classifiers = []
    c = 0
    max_pred = []
    for t in range(500):
        print("*"*10+"for iteration i="+str(i)+" tree number = "+str(t+1)+"feature size = "+str(6)+"*"*10)
        sample_data = random_split_data(train_data,1000,6)
        new_weights = []
        error_t = 0
        algo = ID3()
        sample_data = pd.DataFrame(sample_data)
        answer = dict()
        sample_data.reset_index(drop=True,inplace=True)
        sample_Y_train = sample_data['y']
        all_classifiers.append(algo.ID3_Algo(sample_data,sample_data,list(sample_data.columns[:-1]),labels,split_method,[]))
        if c==0:
            c=1
            for sample in range(X_test.shape[0]):
                sa = dict(X_test.iloc[sample])
                pred = classify(all_classifiers[0],sa)
                single_tree_pred_for_one_tree.append(pred[0])
    single_tree_pred[i] = single_tree_pred_for_one_tree
    if i!=0:
        for sample in range(X_test.shape[0]):
            sa = dict(X_test.iloc[sample])
            pred = groupclassify(all_classifiers,sa)
            multi_tree_pred.append(pred)
    multi_tree_predictions[i] = multi_tree_pred
    
with open('singletreeclassifiers_100_fea_size_6.pickle', 'wb') as handle:
    pickle.dump(single_tree_pred, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('multi_tree_predictions_100_fea_size_6.pickle', 'wb') as handle:
    pickle.dump(multi_tree_predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)

singletrees = []
with (open("singletreeclassifiers_100_fea_size_6.pickle", "rb")) as openfile:
    while True:
        try:
            singletrees.append(pickle.load(openfile))
        except EOFError:
            break

binary_singletrees = dict()
for i in singletrees[0].keys():
    binary_singletrees[i] = [0 if j=='n' or j=='no' else 1 for j in singletrees[0][i]]
singletrees_df=pd.DataFrame.from_dict(binary_singletrees,orient='index').transpose()

multitrees = []
with (open("multi_tree_predictions_100_fea_size_6.pickle", "rb")) as openfile:
    while True:
        try:
            multitrees.append(pickle.load(openfile))
        except EOFError:
            break
binary_multitrees = dict()
for i in multitrees[0].keys():
    binary_multitrees[i] = [0 if j=='n' or j=='no' else 1 for j in multitrees[0][i]]
multitrees_df=pd.DataFrame.from_dict(binary_multitrees,orient='index').transpose()
multitrees_df = multitrees_df.drop(0,axis=1)

ground_truth = []
for i in range(len(Y_test)):
    if Y_test[i]=="no":
        ground_truth.append(0)
    else:
        ground_truth.append(1)
        
average = np.array(singletrees_df.mean(axis=1))
ground_truth = np.array(ground_truth)
bias = np.mean((average-ground_truth)**2)
variance = (1/(len(ground_truth)-1))*sum((ground_truth-average)**2)

print("Bias and Variance of single tree learner is :",(bias,variance))
print("Expected error for single tree learner is ", bias+variance)

average_multi = np.array(multitrees_df.mean(axis=1))
bias = np.mean((average_multi-ground_truth)**2)
variance = (1/(len(ground_truth)-1))*sum((ground_truth-average_multi)**2)
print("Bias and Variance of whole forest is :",(bias,variance))
print("Expected error for whole forest is ", bias+variance)