#In[]
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt

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
        for i in label_data.keys():
            weighted_sum =  sum(data[data['y']==i]['weights'])/sum(data['weights'])
            entropy -= weighted_sum *math.log2(weighted_sum)
        return entropy
    
    def fea_cat_entropy(self,data, feature):
        categories_features = data[feature].value_counts().keys()
        cat_entropy = 0
        total_data_length = len(data)
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['y'].value_counts()
            cat_sum = sum(data[data[feature]==cat]['weights'])
            s = 0
            for i in label_fea_data.keys():
                weighted_sum =  sum(data[(data[feature]==cat) & (data['y']==i)]['weights'])/cat_sum
                s -= (weighted_sum) * math.log2(weighted_sum)
            cat_entropy += (sum(label_fea_data)/total_data_length) * (s)
        return cat_entropy
    
    def total_gini(self,data):
        label_data = data['y'].value_counts()
        entropy = 1
        for i in label_data.keys():
            weighted_sum =  sum(data[data['y']==i]['weights'])/sum(data['weights'])
            entropy -= (weighted_sum)**2
        return entropy
    
    def fea_cat_gini(self,data, feature):
        categories_features = data[feature].value_counts().keys()
        cat_entropy = 0
        total_data_length = len(data)
        for cat in categories_features:
            label_fea_data = data[data[feature]==cat]['y'].value_counts()
            cat_sum = sum(data[data[feature]==cat]['weights'])
            s = 1
            for i in label_fea_data.keys():
                weighted_sum =  sum(data[(data[feature]==cat) & (data['y']==i)]['weights'])/cat_sum
                s -= (weighted_sum)**2
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
        
    def ID3_Algo(self,data,data_copy,features,max_depth,depth,split_method, backup_features=[]):
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
            subtree_dict[i] = self.ID3_Algo(data, data_copy, features,max_depth,depth,split_method, backup_features)
        final_tree = (root_node,subtree_dict)
        last_feature = backup_features.pop()
        features.append(last_feature)
        return final_tree
    
def classify(tree, query):
      labels = ['yes','no']
      if tree in labels:
          return tree
      key = query.get(tree[0])
      if key not in tree[1]:
          key = None
      if key:
        class_ = classify(tree[1][key], query)
      else:
        class_ = "no"
      return class_

#In[]:

train_errors = []
test_errors = []
labels = list(train_data['y'].value_counts().keys())
split_method = "entropy"
max_depth = 2
D = [1/train_data.shape[0]]*train_data.shape[0]
train_data.insert(len(train_data.columns)-1,"weights",D)
alpha_t = []
hypothesis_t = []

for T in range(500):
    algo = ID3()
    train_data["weights"] = D
    h_t = algo.ID3_Algo(train_data, train_data, columns[:-1], max_depth, 1, split_method, [])
    hypothesis_t.append(h_t)

    # Calculate error_t
    predictions = np.array([classify(h_t, dict(sample)) for _,sample in X_train.iterrows()]).squeeze()
    error_t = np.sum(D * (predictions != Y_train))
    
    errot_t = error_t / sum(D)
    print("Error: ",error_t)
    print("Weight Vector: ", D[:10])
    a_t = 0.5 * np.log((1 - errot_t) / errot_t)
    alpha_t.append(a_t)

    # Update weights
    # Vectorized weight update
    D = D * np.exp(np.where(predictions == Y_train, -a_t, a_t))
    D /= np.sum(D)  # Normalize

    # Calculate train error
    train_pred = [[a * (classify(h, dict(sample)) == "yes") for a, h in zip(alpha_t, hypothesis_t)] for _, sample in X_train.iterrows()]
    train_ensemble_predictions = np.sign(np.sum(train_pred,axis=1)).squeeze()
    train_error = np.mean(train_ensemble_predictions != (Y_train == "yes"))
    train_errors.append(train_error)

    # Calculate test error
    
    test_pred = [[a * (classify(h, dict(sample)) == "yes") for a, h in zip(alpha_t, hypothesis_t)] for _, sample in X_test.iterrows()]
    test_ensemble_predictions = np.sign(np.sum(test_pred,axis=1)).squeeze()
    test_error = np.mean(test_ensemble_predictions != (Y_test == "yes"))
    test_errors.append(test_error)

    print(f"Epoch: {T}. Training error: {train_error:.4f}. Test Error: {test_error:.4f}")
# In[116]:

plt.plot(range(len(train_errors)), train_errors, label = "train_error")
plt.plot(range(len(test_errors)), test_errors, label = "test_error")
plt.legend()
plt.ylabel("Error")
plt.xlabel("No. of rounds(T)")
plt.title("train and test errors")
plt.show()







