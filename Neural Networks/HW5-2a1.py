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
        self.weight_1 = np.random.normal(size = (X_train.shape[1],n1))
        self.weight_2 = np.random.normal(size = (self.n1+1,self.n2))
        self.weight_3 = np.random.normal(size = (self.n2+1,1))
    
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
        self.loss = self.loss(self.output_3, y)
        return self.output_3
    def backward(self, output, x, y):
        self.grad_3 = (output-y)*self.output_2
        temp = (output -y)*self.weight_3[1:].T *self.derivative_sigmoid(self.output_2[1:])
        self.grad_2 = (np.repeat(temp,self.n1+1,0).T* self.output_1).T
        temp = np.sum(np.dot((output -y)*self.weight_3[1:]*self.derivative_sigmoid(self.output_2[1:]),self.weight_2[1:].T),axis =0).reshape(1,-1)
        self.grad_1 = ((np.repeat(temp,len(x),0)*self.derivative_sigmoid(self.output_1[1:])).T*x).T
        return self.grad_3, self.grad_2, self.grad_1
    
print("Running problem 2(a)")
nn = NeuralNetwork(int(sys.argv[1]),int(sys.argv[2]))
output = nn.forward(X_train[0],Y_train[0])
param_3, param_2, param_1 = nn.backward(output, X_train[0],Y_train[0])
print("*"*50)
print("Weight param 1")
print(param_1)
print("*"*50)
print("Weight param 2")
print(param_2)
print("*"*50)
print("Weight param 3")
print(param_3)

        
        
        
    