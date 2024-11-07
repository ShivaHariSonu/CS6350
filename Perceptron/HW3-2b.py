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
r=0.001
epochs = 10
X = np.array(X_train)
Y = np.array(Y)

def prediction(x,w):
    return 1 if np.dot(w.T,x)>0 else 0

def predict(X_test,all_weights,C):
    predictions = []
    for x in X_test:
        score = 0
        for w,c in zip(all_weights,C):
            score = score + c*np.sign(np.dot(w,x))
        if score>=0:
            predictions.append(1)
        else:
            predictions.append(0)
    return predictions
k = 0
total_weights = [np.zeros_like(X)[0]]
counts = [0]
epochs = 10
for _ in range(epochs):
    for x,y in zip(X,Y):
        pred = prediction(x,total_weights[k])
        z = -1 if y==0 else 1
        if pred !=y:
            total_weights.append(total_weights[k]+ (r*x*z))
            counts.append(1)
            k+=1
        else:
            counts[k]+=1

pred_test = predict(x_test,total_weights,counts)
teerror = 0
for true,pred in zip(y_test,pred_test):
    if true!=pred:
        teerror+=1

print("The list of distinct weight vectors are as follows ",total_weights)
print("and its respective counts",counts)
print("The number of correctly predicted points are {} and misclassified points are {} and the average test error is {}".format(len(x_test)-teerror,teerror,teerror/len(x_test)))

