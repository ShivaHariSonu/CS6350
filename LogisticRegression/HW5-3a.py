import pandas as pd
import numpy as np
import math
import random
import warnings
warnings.filterwarnings("ignore") 


train_data = pd.read_csv("bank-note/train.csv",header=None)
test_data = pd.read_csv("bank-note/test.csv",header=None)



X = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]
X_train = np.column_stack(([1]*X.shape[0], X))
train_data = np.column_stack(([1]*train_data.shape[0], train_data))
train_data = pd.DataFrame(train_data)
X_test = test_data.iloc[:,:-1]
X_test = np.column_stack(([1]*X_test.shape[0], X_test))
Y_test = test_data.iloc[:,-1]

m,n = X_train.shape


def grad_dw(x,y,w,v):
    grad = ((np.exp(-y*w.T*x)*(-y*x))/(1+np.exp(-y*w.T*x)))+(w/v)
    return grad



learning_rate = 0.001
d = 0.2
epochs = 100
for v in [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]:
    weights = np.random.normal(0,1,size=n)
    for epoch in range(epochs):
        train_data = train_data.sample(frac=1).reset_index(drop=True)
        X_train = train_data.iloc[:,:-1]
        Y_train = train_data.iloc[:,-1]
        for j in range(len(X_train)):
            dw = grad_dw(X_train.iloc[j],Y_train[j],weights,v)
            weights = weights - learning_rate * np.array(dw)
        learning_rate = learning_rate/(1+((learning_rate/d)*epoch))
    
    correct_pred = 0
    for i in range(len(X_train)):
        pred = np.dot(X_train.iloc[i],weights.T)
        if pred <0.5 and Y_train[i]==0:
            correct_pred+=1
        if pred>=0.5 and Y_train[i]==1:
            correct_pred+=1
    print("Train Error {} for setting v {} and accuracy is {}".format((len(X_train)-correct_pred)/len(X_train),v,(correct_pred/len(X_train)*100)))
    correct_pred = 0
    for i in range(len(X_test)):
        pred = np.dot(X_test[i],weights.T)
        if pred<0.5 and Y_test[i]==0:
            correct_pred+=1
        if pred>=0.5 and Y_test[i]==1:
            correct_pred+=1
    print("Test Error {} for setting v {} and accuracy is {}".format((len(X_test)-correct_pred)/len(X_test),v,(correct_pred/len(X_test)*100)))

