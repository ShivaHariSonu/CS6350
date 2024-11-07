

import numpy as np
import pandas as pd
import math

train_data = pd.read_csv("bank-note/train.csv",header=None)
test_data = pd.read_csv("bank-note/test.csv",header=None)

X = train_data.iloc[:,:-1]
Y = train_data.iloc[:,-1]

X_train = np.column_stack(([1]*X.shape[0], X)) # adding bias component for train and test
x_test = test_data.iloc[:,:-1]
x_test = np.column_stack(([1]*x_test.shape[0], x_test))
y_test = test_data.iloc[:,-1]

m,n = X_train.shape
weights = np.zeros(n)
a = np.zeros(n)
r=0.001
epochs = 10
X = np.array(X_train)
Y = np.array(Y)

def prediction(x,w):
    return 1 if np.dot(w.T,x)>0 else 0

for _ in range(epochs):
    for x,y in zip(X,Y):
        pred = prediction(x,weights)
        z = -1 if y==0 else 1
        if pred !=y:
            weights = weights + (r*x*z)
        a = a+weights
    

error = 0
for x,y in zip(x_test,y_test):
    pred_test = prediction(x,a/len(X))
    if pred_test!=y:
        error +=1
print("The learned average weight vector is ",a)
print("The Average Learned weight vector is ",a/len(X))
print("For learning rate {} the error is {} and the average prediction error {}".format(r,error,error/len(x_test)))