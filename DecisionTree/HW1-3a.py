#In[1]:
import numpy as np
import math
import pandas as pd

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


def classify(tree, query):
      if tree in labels:
          return tree
      key = query.get(tree[0])
      if key not in tree[1]:
          key = None
      class_ = classify(tree[1][key], query)
      return class_



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
    
    def find_bestsplits(self,data,data_copy,max_depth,depth,root_node):
        features_groups = dict()
        temp_node_values = list(data[root_node].value_counts().keys())
        root_node_values = list(data_copy[root_node].value_counts().keys())
        if depth<max_depth:
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
          return features_groups,depth+1
        else:
          for i in root_node_values:
                if i in temp_node_values:
                  label_counts = dict(data[data[root_node]==i]['y'].value_counts())
                  features_groups[i] =  [({}, max(label_counts,key=label_counts.get))]
                else:
                  label_counts = dict(data['y'].value_counts())
                  features_groups[i] =  [({}, max(label_counts,key=label_counts.get))]
          return features_groups,depth
        
    def ID3_Algo(self,data,data_copy,features,max_depth,depth,split_method,backup_features=[]):
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
        feature_groups,depth = self.find_bestsplits(data, data_copy,max_depth,depth,root_node)
        subtree_dict = {}
        for i ,j in feature_groups.items():
            data = pd.DataFrame.from_dict({k: dict(v) for k,v in pd.DataFrame(j)[0].items()}, orient='index')
            data['y'] = pd.DataFrame(j)[1]
            subtree_dict[i] = self.ID3_Algo(data, data_copy, features,max_depth,depth,split_method,backup_features)
        final_tree = (root_node,subtree_dict)
        last_feature = backup_features.pop()
        features.append(last_feature)
        return final_tree


train_error = dict()
test_error = dict()

labels = list(train_data['y'].value_counts().keys())

for max_depth in range(1,17):
    for split_method in ["entropy","gini","majorityerror"]:
        print("*"*10+" Training Decision Tree with depth "+str(max_depth)+" and split method "+split_method+" "+"*"*10)
        algo = ID3()
        answer = dict()
        s = algo.ID3_Algo(train_data,train_data,columns[:-1],max_depth,1,split_method,[])
        c=0
        for i in range(X_train.shape[0]):
            sample = dict(X_train.iloc[i])
            if classify(s,sample)[0]==Y_train[i]:
                c+=1
        print("Train missclassified points: ",(X_train.shape[0]-c)) 
        print("Training Error: ",(X_train.shape[0]-c)/X_train.shape[0])
        if train_error.get(max_depth):
            train_error[max_depth].append((split_method,(X_train.shape[0]-c)/X_train.shape[0]))
        else:
            train_error[max_depth] = [(split_method,(X_train.shape[0]-c)/X_train.shape[0])]
        c=0
        for i in range(X_test.shape[0]):
            sample = dict(X_test.iloc[i])
            if classify(s,sample)[0]==Y_test[i]:
                c+=1
        print("test missclassified points: ",(X_test.shape[0]-c))
        print("Testing Error: ",(X_test.shape[0]-c)/X_test.shape[0])
        if test_error.get(max_depth):
            test_error[max_depth].append((split_method,(X_test.shape[0]-c)/X_test.shape[0]))
        else:
            test_error[max_depth] = [(split_method,(X_test.shape[0]-c)/X_test.shape[0])]
        print("*"*70)


# In[4]:
train_ans = []
k = 0
for i in train_error.keys():
    train_ans.append(["Depth = "+ str(i)])
    for j in range(len(train_error[i])):
        train_ans[k].append(train_error[i][j][1])
    k+=1
test_ans = []
k = 0
for i in test_error.keys():
    test_ans.append(["Depth = "+str(i)])
    for j in range(len(test_error[i])):
        test_ans[k].append(test_error[i][j][1])
    k+=1

# In[2]:
from prettytable import PrettyTable
x = PrettyTable()
x.field_names = ["depth \ split_method","Entropy", "Gini Index", "Majority Error"]
x.add_rows(train_ans)
print(x)

# In[3]:
x = PrettyTable()
x.field_names = ["depth \ split_method","Entropy", "Gini Index", "Majority Error"]
x.add_rows(test_ans)
print(x)

# %%
