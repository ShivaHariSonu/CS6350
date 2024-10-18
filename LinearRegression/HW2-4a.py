#In[]:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import random


train_data = pd.read_csv("concrete/train.csv",header=None)
test_data = pd.read_csv("concrete/test.csv",header=None)
X = train_data.iloc[:,:-1]
Y = train_data.iloc[:,-1]
X_train = np.column_stack(([1]*X.shape[0], X)) # added the bias
m,n = X_train.shape


weights = np.array([0]*n)
X = np.array(X_train)
Y = np.array(Y)
r = 0.006
iterations = 1000
cost = []
weights_new = []


# In[234]:


predictions_at_step = []


def gradient_dw(x,y,w,grad_dw):
    for i in range(len(x)):
        pred = np.matmul(w.T,x)
        grad_dw[i] = -(y-pred)*x[i]
    return grad_dw
step=0
dw = np.array([0]*n)
gradient_update = np.array([0]*n)
while 1:
    for j in range(len(X)):
        dw = gradient_dw(X[j],Y[j],weights,dw)
        gradient_update = gradient_update + dw
    predictions_at_step.append((1/2)*sum(Y - np.matmul(weights,X.T)**2))
    weights_new = weights - r * np.array(gradient_update)
    print("The decaying learning rate over the step {} is {}".format(step+1,r))
    r = r/2
    if np.count_nonzero(abs(weights_new - weights)<10**-6)==len(weights):
        break
    weights = weights_new
    step+=1
#In[]:
print("The weights that are learned: ", weights)
print("In this the bias is : ",weights[0])

#In[]:
X_t = test_data.iloc[:,:-1]
Y_t = test_data.iloc[:,-1]
X_test = np.column_stack(([1]*X_t.shape[0], X_t))



# In[]
def lms(true,pred):
    return (1/len(true))*sum((true-pred)**2)

pred = np.matmul(X,weights)
print("The cost function for the train data is:",lms(Y,pred))


pred = np.matmul(X_test,weights)
print("The cost function for the test data is:",lms(Y_t,pred))


#In[]

plt.plot(range(len(predictions_at_step)),predictions_at_step)
plt.xlabel("Epochs")
plt.ylabel("Cost value")
plt.title("Cost function changes along with steps")
plt.show()
# %%
