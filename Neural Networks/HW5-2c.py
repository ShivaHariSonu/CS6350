import pandas as pd
import numpy as np
import math
import random
import sys



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

class NeuralNetwork:
    def __init__(self,n1,n2):
        self.n1 = n1
        self.n2 = n2
        self.weight_1 = np.zeros((X_train.shape[1],n1))
        self.weight_2 = np.zeros((self.n1+1,self.n2))
        self.weight_3 = np.zeros((self.n2+1,1))
    
    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def derivative_sigmoid(self, z):
        return z*(1-z)
    
    def loss(self, y_pred, y):
        return (0.5)*(y_pred-y)**2
    
    def forward(self, x, y):
        self.output_1 = np.append(1, self.sigmoid(np.dot(x,self.weight_1)))
        self.output_2 = np.append(1, self.sigmoid(np.dot(self.output_1, self.weight_2)))
        self.output_3 = np.dot(self.output_2,self.weight_3)
        loss_value = self.loss(self.output_3, y)
        return self.output_3
    
    def backward(self, output, x, y):
        self.grad_3 = (output-y)*self.output_2
        temp = (output -y)*self.weight_3[1:].T *self.derivative_sigmoid(self.output_2[1:])
        self.grad_2 = (np.repeat(temp,self.n1+1,0).T* self.output_1).T
        temp = np.sum(np.dot((output -y)*self.weight_3[1:]*self.derivative_sigmoid(self.output_2[1:]),self.weight_2[1:].T),axis =0).reshape(1,-1)
        self.grad_1 = ((np.repeat(temp,len(x),0)*self.derivative_sigmoid(self.output_1[1:])).T*x).T
        return self.grad_3, self.grad_2, self.grad_1

    def fit(self, train_data, X_train, Y_train, X_test, Y_test, lr_0 = 0.01,epochs =10):
        lr = lr_0
        for epoch in range(1,epochs+1):
            train_data = train_data.sample(frac=1).reset_index(drop= True)
            X = train_data.iloc[:,:-1]
            y = train_data.iloc[:,-1]
            for i in range(len(X)):
                output = self.forward(X.iloc[i].values,y[i])
                grad3, grad2, grad1  = self.backward(output, X.iloc[i].values,y[i])
                self.weight_1 = self.weight_1 - grad1*lr
                self.weight_2 = self.weight_2 - grad2*lr
                self.weight_3 = (self.weight_3.reshape(1,-1) - lr*grad3.reshape(1,-1)).reshape(-1,1)
            lr = lr_0/(1+((lr_0*epoch)/0.1))
        train_pred = []
        test_pred = []
        for x,y in zip(X_train,Y_train):
            train_pred.append(self.forward(x,y))
        for x,y in zip(X_test,Y_test):
            test_pred.append(self.forward(x,y))
        return train_pred, test_pred
    
    

print("Running problem 2b")
neurons = [5,10,25,50,100]
for n in neurons:
    nn = NeuralNetwork(n,n)
    train_pred, test_pred = nn.fit(train_data, X_train, Y_train, X_test, Y_test,lr_0 = 0.1, epochs=50)
    count = 0
    print("Number of hidden neurons (width) {}".format(n))
    for i in range(len(train_pred)):
        if (train_pred[i]>0.5 and Y_train[i]==1):
            count+=1
        if (train_pred[i]<0.5 and Y_train[i]==0):
            count+=1
    print("Training error is {}. Training accuracy is {}".format((len(train_pred)-count)/len(train_pred), (count/len(train_pred))*100 ))
    
    count = 0
    for i in range(len(test_pred)):
        if (test_pred[i]>0.5 and Y_test[i]==1):
            count+=1
        if (test_pred[i]<0.5 and Y_test[i]==0):
            count+=1
    print("Test error is {}. Test accuracy is {}".format((len(test_pred)-count)/len(test_pred), (count/len(test_pred))*100 ))
    print("*"*50)
    

        
        
        
    