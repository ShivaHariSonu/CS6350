#In[]
import numpy as np
import pandas as pd

train_data = pd.read_csv("concrete/train.csv",header=None)

X = train_data.iloc[:,:-1]
Y = train_data.iloc[:,-1]
X_train = np.column_stack(([1]*X.shape[0], X)) # adding bias feature
X = np.array(X_train)
Y = np.array(Y)

print("Optimal weight vector that is calculated with the analytical form is : ",np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y)))

#In[]:

X = train_data.iloc[:,:-1]
Y = train_data.iloc[:,-1]
X = np.array(X)
Y = np.array(Y)

print("Optimal weight vector that is calculated with the analytical form without bias feature is : ",np.dot(np.linalg.inv(np.dot(X.T,X)),np.dot(X.T,Y)))

# %%
