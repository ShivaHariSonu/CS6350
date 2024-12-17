#In[]:
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math


data = pd.read_excel("credit_card/default of credit card clients.xls")
data = data.drop("Unnamed: 0",axis=1)
data.columns = data.iloc[0]
Y = data['default payment next month']
data = data.drop("default payment next month",axis=1)
data = data.drop(0).reset_index(drop=True)
normalized_data=(data-data.mean())/data.std()


train_indexs = random.sample(range(normalized_data.shape[0]),k=24000)
test_indexs = list(set(range(normalized_data.shape[0])) - set(train_indexs))

train_data = normalized_data.iloc[train_indexs].reset_index(drop=True)
Y_train = Y[train_indexs].reset_index(drop=True)
test_data = normalized_data.iloc[test_indexs].reset_index(drop=True)
Y_test = Y[test_indexs].reset_index(drop=True)
train_data['y'] = Y_train
test_data['y'] = Y_test


X_train = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]

X_test = test_data.iloc[:,:-1]
Y_test = test_data.iloc[:,-1]
#In[]
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


#In[]:

def classify(tree, query):
    labels = ['yes','no']
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



def random_forest(trees,fea_size):
    all_classifiers = []
    for t in range(trees):
        print("*"*10+"trees = "+str(t+1)+" for feature sizes = "+str(fea_size)+"*"*10)
        sample_data = random_split_data(train_data,train_data.shape[0],fea_size)
        algo = ID3()
        sample_data = pd.DataFrame(sample_data)
        all_classifiers.append(algo.ID3_Algo(sample_data,sample_data,list(sample_data.columns[:-1]),split_method,[]))
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


#In[]

labels = ["yes","no"]
split_method = "entropy"
print("Random Forest")
train_error = []
train_error_fea = dict()
test_error_fea = dict()
test_error = []
for fea_size in [6]:
    for i in range(1,501):
        train_e,test_e = random_forest(i,fea_size)
        train_error.append(train_e)
        test_error.append(test_e)
        print(train_error,test_error)
    train_error_fea[fea_size] = train_error
    test_error_fea[fea_size] = test_error
print("train errors: ",train_error_fea)
print("test errors: ",test_error_fea)

#In[]

file = open("train_errors.txt", "w+")
content = str(train_error)
file.write(content)
file.close()
file = open("test_errors.txt", "w+")
content = str(test_error)
file.write(content)
file.close()

samplesize = 5000


def bagging(trees):
    all_classifiers = []
    for t in range(trees):
        algo = ID3()
        train_data_sampled = train_data.sample(samplesize,replace=True)
        h_t = algo.ID3_Algo(train_data_sampled ,train_data_sampled ,list(train_data_sampled.columns[:-1]),split_method,[])
        all_classifiers.append(h_t)
    c= 0
    for i in range(X_train.shape[0]):
        sample = dict(X_train.iloc[i])
        if groupclassify(all_classifiers,sample)==Y_train[i]:
            c+=1
    train_error = (X_train.shape[0]-c)/X_train.shape[0]

    c= 0
    for i in range(X_test.shape[0]):
        sample = dict(X_test.iloc[i])
        if groupclassify(all_classifiers,sample)==Y_test[i]:
            c+=1
    test_error = (X_test.shape[0]-c)/X_test.shape[0]
    return train_error, test_error


print("Bagging")
train_error = []
test_error = []
for i in range(1,501):
    train_e,test_e = bagging(i)
    train_error.append(train_e)
    test_error.append(test_e)
    print(train_error,test_error)
    
    
    
    
    
    
    
    
    
    
print("AdaBoost")
class ID3ForAdaBoost:
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
                weighted_sum =  sum(data[data[feature]==cat and data['y']==i]['weights'])/cat_sum
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
                weighted_sum =  sum(data[data[feature]==cat and data['y']==i]['weights'])/cat_sum
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
        
    def ID3_Algo(self,data,data_copy,features,max_depth,depth,split_method, weights, backup_features=[]):
        if len(data['y'].value_counts()) == 1:
            return data['y'].value_counts().keys()
        if len(features)==0: 
            label_counts = dict(data['y'].value_counts())
            return max(label_counts,key=label_counts.get)
        
        data['weights'] = weights
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
            subtree_dict[i] = self.ID3_Algo(data, data_copy, features,max_depth,depth,split_method, weights, backup_features)
        final_tree = (root_node,subtree_dict)
        last_feature = backup_features.pop()
        features.append(last_feature)
        return final_tree
    


all_weights = []
vote_alpha = []
D = [1/train_data.shape[0]] * train_data.shape[0]
all_weights.append(D)
all_weak_classifiers = []
train_errors = []
test_errors = []
all_errors = []
for t in range(500):
    print("*"*10+"Iteration {}".format(t+1)+"*"*10)
    new_weights = []
    error_t = 0
    algo = ID3ForAdaBoost()
    labels = ['yes','no']
    answer = dict()
    max_depth = 1
    split_method = "entropy"
    all_weak_classifiers.append(algo.ID3_Algo(train_data,train_data,list(train_data.columns[:-1]),max_depth,1,split_method,all_weights[t],[]))
    print(all_weak_classifiers)
    c=0
    for i in range(X_train.shape[0]):
      sample = dict(X_train.iloc[i])
      if classify(all_weak_classifiers[t],sample)!=Y_train[i]:
            error_t += train_data['weights'][i]*1
      else:
            c+=1
    error_t = error_t/sum(train_data['weights'])
    print(error_t)
    vote_alpha.append((1/2)*(math.log(1-error_t)/error_t))
    for i in range(X_train.shape[0]):
        sample = dict(X_train.iloc[i])
        if classify(all_weak_classifiers[t],sample)!=Y_train[i]:
            new_weights.append(all_weights[t][i] * math.exp(vote_alpha[t]))
        else:
            new_weights.append(all_weights[t][i] * math.exp(-vote_alpha[t]))
    all_weights.append(new_weights)
    train_errors.append((X_train.shape[0]-c)/X_train.shape[0])
    c_test=0
    for i in range(X_test.shape[0]):
      sample = dict(X_test.iloc[i])
      if classify(all_weak_classifiers[t],sample)==Y_test[i]:
        c_test+=1
      else:
        if classify(all_weak_classifiers[t],sample)=="y" or classify(all_weak_classifiers[t],sample)=="n":
            print(classify(all_weak_classifiers[t],sample))
    test_errors.append((X_test.shape[0]-c_test)/X_test.shape[0])
    all_weights[t+1] = (np.array(all_weights[t+1]) / sum(all_weights[t+1]))
    print(train_errors,test_errors)

