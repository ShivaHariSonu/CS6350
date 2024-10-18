#In[1]:
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
  
def random_split_data(train_data,samplesize,fea_size):
    sa = random.choices(range(train_data.shape[0]),k=samplesize)
    row_samples = train_data.iloc[sa].reset_index(drop=True)
    samples = random.sample(range(len(train_data.columns)-1),k=fea_size)
    #Adding the label column
    samples.append(16)
    sampled_data = row_samples.iloc[:,samples]
    return sampled_data

def groupclassify(all_classifiers,sample):
    pred = []
    for classifier in all_classifiers:
        if classify(classifier,sample)=="yes":
            pred.append("yes")
        else:
            pred.append("no")
    return max(set(pred), key = pred.count)

def random_forest(trees,fea_size):
    all_classifiers = []
    for t in range(trees):
        print("*"*10+"trees = "+str(t+1)+" for feature sizes = "+str(fea_size)+"*"*10)
        sample_data = random_split_data(train_data,train_data.shape[0],fea_size)
        algo = ID3()
        sample_data = pd.DataFrame(sample_data)
        all_classifiers.append(algo.ID3_Algo(sample_data,sample_data,list(sample_data.columns[:-1]),labels,split_method,[]))
    final_pred = []
    for i in range(X_train.shape[0]):
        final_pred.append(groupclassify(all_classifiers,X_train.iloc[i]))
    c_train = 0
    for i in range(len(final_pred)):
        if final_pred[i]==Y_train[i]:
            c_train+=1
    final_pred_test = []
    for i in range(X_test.shape[0]):
        final_pred_test.append(groupclassify(all_classifiers,X_test.iloc[i]))
    c_test = 0
    for i in range(len(final_pred_test)):
        if final_pred_test[i]==Y_test[i]:
            c_test+=1
    return (X_train.shape[0]-c_train)/X_train.shape[0],(X_test.shape[0]-c_test)/X_test.shape[0]

split_method = "entropy"
labels = ['yes','no']

for feature_size in [2,4,6]:
    train_error=[]
    test_error=[]
    for trees in range(1,501):
        train_error_t,test_error_t = random_forest(trees,feature_size)
        train_error.append(train_error_t)
        test_error.append(test_error_t)
        print(train_error_t,test_error_t)
    plt.plot(train_error,label="train_error")
    plt.plot(test_error,label="test_error")
    plt.legend()
    plt.xlabel("Number of Trees")
    plt.ylabel("Train and Test Errors")
    plt.title("Train and Test Error for Feature size = "+str(feature_size))
    plt.show()