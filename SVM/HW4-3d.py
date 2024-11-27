import numpy as np
import pandas as pd
import math
import random
import matplotlib.pyplot as plt
from scipy.optimize import minimize,Bounds
import warnings
warnings.filterwarnings("ignore")


train_data = pd.read_csv("bank-note/train.csv",header=None)
test_data = pd.read_csv("bank-note/test.csv",header=None)

def labels_convert(value):
    return -1 if value==0 else 1

train_data.iloc[:,-1] = train_data.iloc[:,-1].map(labels_convert)
test_data.iloc[:,-1] = test_data.iloc[:,-1].map(labels_convert)


X = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]
X_train = np.column_stack(([1]*X.shape[0], X))
train_data = np.column_stack(([1]*train_data.shape[0], train_data))
train_data = pd.DataFrame(train_data)
X_test = test_data.iloc[:,:-1]
X_test = np.column_stack(([1]*X_test.shape[0], X_test))
Y_test = test_data.iloc[:,-1]

def gaussian_kernel(x, y, gamma):
    return np.exp(-np.sum(np.square(x - y)) / gamma)

def prediction(kernel,x1,x2,y,count,ga):
    pred = np.sum(count*y*kernel(x1,x2,ga))
    return 1 if pred>0 else -1
    
lr = [10**x for x in range(-5,5)]
c = [0]*len(X_train)
gamma = [0.1, 0.5, 1, 5, 100]
for g in gamma:
    print("Gamma value as {}".format(g))
    for _ in range(100):
        i = 0
        for x,y in zip(X_train,Y_train):
            pred = prediction(gaussian_kernel,X_train,x,y,np.array(c),g)
            if pred!=y:
                    c[i] +=1
            i+=1
    error = 0
    for x,y in zip(X_train,Y_test):
        pred_test = prediction(gaussian_kernel,X_train,x,y,np.array(c),g)
        if pred_test!=y:
            error +=1
    print("Train error:- {}. Train accuracy:- {}".format(error/len(X_train),(len(X_train)-error)/len(X_train)))
    error = 0
    for x,y in zip(X_test,Y_test):
        pred_test = prediction(gaussian_kernel,X_train,x,y,np.array(c),g)
        if pred_test!=y:
            error +=1
    print("Test error:- {}. Test accuracy:- {}".format(error/len(X_test),(len(X_test)-error)/len(X_test)))