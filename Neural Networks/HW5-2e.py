
# In[]:
import tensorflow as tf
import pandas as pd
import numpy as np
import math
import random
import warnings
warnings.filterwarnings("ignore") 


train_data = pd.read_csv("bank-note/train.csv",header=None)
test_data = pd.read_csv("bank-note/test.csv",header=None)

X_train = train_data.iloc[:,:-1]
Y_train = train_data.iloc[:,-1]
X_test = test_data.iloc[:,:-1]
Y_test = test_data.iloc[:,-1]

xavier_init = tf.keras.initializers.GlorotNormal()
he_init = tf.keras.initializers.HeNormal()
width = [5,10,25,50,100]

#In[]

for w in width:
    model_tanh = tf.keras.Sequential([
        tf.keras.Input(shape = (4,)),
        tf.keras.layers.Dense(units=w,activation='tanh',kernel_initializer = xavier_init),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=1,activation='linear')
    ])
    model_relu = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(units=w,activation='relu',kernel_initializer=he_init),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=1,activation='linear')
    ])
    model_tanh.compile(optimizer = 'adam', loss= tf.losses.MeanSquaredError(), metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy",threshold=0.5,dtype=None)])
    model_relu.compile(optimizer = 'adam', loss= tf.losses.MeanSquaredError(), metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy",threshold=0.5,dtype=None)])
    
    history = model_tanh.fit(
        X_train,Y_train, epochs=10,validation_data = (X_test,Y_test),verbose=0
    )
    total_pred = model_tanh.predict(X_train,verbose=0)
    correct_pred=0
    for i in range(len(total_pred)):
        if (total_pred[i]>0.5 and Y_train[i]==1):
            correct_pred+=1
        if (total_pred[i]<=0.5 and Y_train[i]==0):
            correct_pred+=1
    print("For Model Tanh")
    print("Train error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("tanh","Xavier Normal",w,3,(len(Y_train)-correct_pred)/len(Y_train)))
    total_pred = model_tanh.predict(X_test,verbose=0)
    correct_pred=0
    for i in range(len(total_pred)):
        if (total_pred[i]>0.5 and Y_test[i]==1):
            correct_pred+=1
        if (total_pred[i]<=0.5 and Y_test[i]==0):
            correct_pred+=1

    print("Test error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("tanh","Xavier Normal", w,3,(len(Y_test)-correct_pred)/len(Y_test)))

    history = model_relu.fit(
        X_train,Y_train, 
        epochs=100, 
        validation_data=(X_test,Y_test),verbose=0
    )
    total_pred = model_relu.predict(X_train,verbose=0)
    correct_pred=0
    for i in range(len(total_pred)):
        if (total_pred[i]>0.5 and Y_train[i]==1):
            correct_pred+=1
        if (total_pred[i]<=0.5 and Y_train[i]==0):
            correct_pred+=1
    print("For Model Relu")
    print("Train error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("ReLu","He Normal",w,3,(len(Y_train)-correct_pred)/len(Y_train)))
    total_pred = model_relu.predict(X_test,verbose=0)
    correct_pred=0
    for i in range(len(total_pred)):
        if (total_pred[i]>0.5 and Y_test[i]==1):
            correct_pred+=1
        if (total_pred[i]<=0.5 and Y_test[i]==0):
            correct_pred+=1

    print("Test error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("ReLu","He Normal", w,3,(len(Y_test)-correct_pred)/len(Y_test)))
    print("*"*50)

    
#In[]:

for w in width:
    model_tanh = tf.keras.Sequential([
        tf.keras.Input(shape = (4,)),
        tf.keras.layers.Dense(units=w,activation='tanh',kernel_initializer = xavier_init),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=1,activation='linear')
    ])
    model_relu = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(units=w,activation='relu',kernel_initializer=he_init),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=1,activation='linear')
    ])
    model_tanh.compile(optimizer = 'adam', loss= [tf.losses.MeanSquaredError(), metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy",threshold=0.5,dtype=None)])
    model_relu.compile(optimizer = 'adam', loss= [tf.losses.MeanSquaredError(), metrics=tf.keras.metrics.BinaryAccuracy(name="binary_accuracy",threshold=0.5,dtype=None)])
    
    history = model_tanh.fit(
        X_train,Y_train, epochs=100,validation_data = (X_test,Y_test),verbose=0
    )
    total_pred = model_tanh.predict(X_train,verbose=0)
    correct_pred=0
    for i in range(len(total_pred)):
        if (total_pred[i]>0.5 and Y_train[i]==1):
            correct_pred+=1
        if (total_pred[i]<=0.5 and Y_train[i]==0):
            correct_pred+=1
    print("For model Tanh")
    print("Train error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("tanh","Xavier Normal",w,5,(len(Y_train)-correct_pred)/len(Y_train)))
    total_pred = model_tanh.predict(X_test,verbose=0)
    correct_pred=0
    for i in range(len(total_pred)):
        if (total_pred[i]>0.5 and Y_test[i]==1):
            correct_pred+=1
        if (total_pred[i]<=0.5 and Y_test[i]==0):
            correct_pred+=1

    print("Test error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("tanh","Xavier Normal", w,5,(len(Y_test)-correct_pred)/len(Y_test)))

    history = model_relu.fit(
        X_train,Y_train, 
        epochs=100, 
        validation_data=(X_test,Y_test),verbose=0
    )
    total_pred = model_relu.predict(X_train,verbose=0)
    correct_pred=0
    for i in range(len(total_pred)):
        if (total_pred[i]>0.5 and Y_train[i]==1):
            correct_pred+=1
        if (total_pred[i]<=0.5 and Y_train[i]==0):
            correct_pred+=1
    print("For model ReLU")
    print("Train error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("ReLu","He Normal",w,5,(len(Y_train)-correct_pred)/len(Y_train)))
    total_pred = model_relu.predict(X_test,verbose=0)
    correct_pred=0
    for i in range(len(total_pred)):
        if (total_pred[i]>0.5 and Y_test[i]==1):
            correct_pred+=1
        if (total_pred[i]<=0.5 and Y_test[i]==0):
            correct_pred+=1

    print("Test error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("ReLu","He Normal", w,5,(len(Y_test)-correct_pred)/len(Y_test)))
    print("*"*50)


#In[]:

for w in width:
    model_tanh = tf.keras.Sequential([
        tf.keras.Input(shape = (4,)),
        tf.keras.layers.Dense(units=w,activation='tanh',kernel_initializer = xavier_init),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=w,activation='tanh'),
        tf.keras.layers.Dense(units=1,activation='linear')
    ])
    model_relu = tf.keras.Sequential([
        tf.keras.Input(shape=(4,)),
        tf.keras.layers.Dense(units=w,activation='relu',kernel_initializer=he_init),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=w,activation='relu'),
        tf.keras.layers.Dense(units=1,activation='linear')
    ])
    model_tanh.compile(optimizer = 'adam', loss= tf.losses.MeanSquaredError(), metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy",threshold=0.5,dtype=None)])
    model_relu.compile(optimizer = 'adam', loss= tf.losses.MeanSquaredError(), metrics=[tf.keras.metrics.BinaryAccuracy(name="binary_accuracy",threshold=0.5,dtype=None)])
    
    history = model_tanh.fit(
        X_train,Y_train, epochs=100,validation_data = (X_test,Y_test),verbose=0
    )
    total_pred = model_tanh.predict(X_train,verbose=0)
    correct_pred=0
    for i in range(len(total_pred)):
        if (total_pred[i]>0.5 and Y_train[i]==1):
            correct_pred+=1
        if (total_pred[i]<=0.5 and Y_train[i]==0):
            correct_pred+=1
    print("For model Tanh")
    print("Train error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("tanh","Xavier Normal",w,9,(len(Y_train)-correct_pred)/len(Y_train)))
    total_pred = model_tanh.predict(X_test,verbose=0)
    correct_pred=0
    for i in range(len(total_pred)):
        if (total_pred[i]>0.5 and Y_test[i]==1):
            correct_pred+=1
        if (total_pred[i]<=0.5 and Y_test[i]==0):
            correct_pred+=1

    print("Test error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("tanh","Xavier Normal", w,9,(len(Y_test)-correct_pred)/len(Y_test)))

    history = model_relu.fit(
        X_train,Y_train, 
        epochs=100, 
        validation_data=(X_test,Y_test),verbose=0
    )
    total_pred = model_relu.predict(X_train,verbose=0)
    correct_pred=0
    for i in range(len(total_pred)):
        if (total_pred[i]>0.5 and Y_train[i]==1):
            correct_pred+=1
        if (total_pred[i]<=0.5 and Y_train[i]==0):
            correct_pred+=1
    print("For model ReLU")
    print("Train error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("ReLu","He Normal",w,9,(len(Y_train)-correct_pred)/len(Y_train)))
    total_pred = model_relu.predict(X_test,verbose=0)
    correct_pred=0
    for i in range(len(total_pred)):
        if (total_pred[i]>0.5 and Y_test[i]==1):
            correct_pred+=1
        if (total_pred[i]<=0.5 and Y_test[i]==0):
            correct_pred+=1

    print("Test error for neural network implementation with Activation {}, weight initialization {}, width {} and depth {} is {}".format("ReLu","He Normal", w,9,(len(Y_test)-correct_pred)/len(Y_test)))
    print("*"*50)


# %%
