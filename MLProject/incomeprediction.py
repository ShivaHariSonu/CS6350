
#In[]
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import  StandardScaler,MinMaxScaler,OneHotEncoder
from sklearn.linear_model import LinearRegression,LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import xgboost as xgb
from sklearn.svm import SVC
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import tensorflow as tf
import warnings
warnings.filterwarnings("ignore")



train_data = pd.read_csv("datasets/train_final.csv")
test_data = pd.read_csv("datasets/test_final.csv")

#Preprocessing the data

#In[]
missing_features = []
for i in train_data.columns:
    count = sum(train_data[i]=='?')
    if count>0:
        missing_features.append(i)
print(missing_features)

#In[]
def majority_features(feature):
    yes_label = dict(train_data[train_data['income>50K']==1][feature].value_counts())
    no_label = dict(train_data[train_data['income>50K']==0][feature].value_counts())
    majority_fea_yes_label = sorted(yes_label,key=yes_label.get,reverse=True)[0]
    majority_fea_no_label = sorted(no_label,key=no_label.get,reverse=True)[0]
    return majority_fea_yes_label, majority_fea_no_label

for feature in missing_features:
    majority_features_yes, majority_features_no = majority_features(feature)
    print("For the feature "+feature+". Majority is yes for "+majority_features_yes+". Majority is no for "+majority_features_no)
    yes_attr_data = train_data[train_data['income>50K']==1]
    no_attr_data = train_data[train_data['income>50K']==0]
    yes_attr_data[feature].replace('?',majority_features_yes,inplace=True)
    no_attr_data[feature].replace('?',majority_features_no,inplace=True)
    train_data = pd.concat([yes_attr_data,no_attr_data])
    train_data = train_data.sample(frac=1).reset_index(drop=True)
    
#In[]
train_data

#In[]
sns.set_style("darkgrid")

# Customize the plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=train_data, x="age", y="education.num", hue="income>50K", 
                palette="viridis", alpha=0.7)

plt.title("Age vs Education Number by Income", fontsize=16)
plt.xlabel("Age", fontsize=12)
plt.ylabel("Education Number", fontsize=12)
plt.legend(title="Income > 50K", title_fontsize='12', fontsize='10')

# Adjust layout and display
plt.tight_layout()
plt.show()

#In[]
sns.set_style("darkgrid");
sns.FacetGrid(train_data, hue="income>50K", height=4) \
   .map(plt.scatter, "hours.per.week", "education.num") \
   .add_legend();
plt.show();

#In[]
sns.set_style("darkgrid");
sns.FacetGrid(train_data, hue="income>50K", height=4) \
   .map(plt.scatter, "hours.per.week", "age") \
   .add_legend();
plt.show();
    
    
#In[]
Y_train_data = train_data['income>50K']
X_train_data = train_data.drop('income>50K',axis=1)

#In[]
#As we see the majority is Private for work class for both yes and no label.
#And native country is max for United States for both cases.
# We take feature occupation is max for Other Service.
test_data['workclass'] = test_data['workclass'].replace('?','Private')
test_data['occupation'] = test_data['occupation'].replace('?','Other-service')
test_data['native.country'] = test_data['native.country'].replace('?','United-States')

#In[]
ID = test_data['ID']
X_test = test_data.drop('ID',axis=1)
X_train,X_cv,Y_train,Y_cv = train_test_split(X_train_data,Y_train_data,test_size=0.15,random_state=42,stratify=Y_train_data)


#In[]
def preprocess_categorical(X_train, X_cv, X_test, column):
    vectorizer = CountVectorizer()
    train_pre = vectorizer.fit_transform(X_train[column].values.astype('U'))
    cv_pre = vectorizer.transform(X_cv[column].values.astype('U'))
    test_pre = vectorizer.transform(X_test[column].values.astype('U'))
    return train_pre, cv_pre, test_pre

def preprocess_numerical(X_train, X_cv, X_test, column):
    scaler = StandardScaler()
    train_pre = scaler.fit_transform(X_train[column].values.reshape(-1, 1))
    cv_pre = scaler.transform(X_cv[column].values.reshape(-1, 1))
    test_pre = scaler.transform(X_test[column].values.reshape(-1, 1))
    return train_pre, cv_pre, test_pre

# Categorical columns
cat_columns = ['workclass', 'education', 'marital.status', 'occupation', 'relationship', 'race', 'native.country']
cat_features = [preprocess_categorical(X_train, X_cv, X_test, col) for col in cat_columns]

# Special case for 'sex' column
sex_encoder = OneHotEncoder()
sex_features = (
    sex_encoder.fit_transform(X_train['sex'].values.reshape(-1, 1)),
    sex_encoder.transform(X_cv['sex'].values.reshape(-1, 1)),
    sex_encoder.transform(X_test['sex'].values.reshape(-1, 1))
)

# Numerical columns
num_columns = ['age', 'fnlwgt', 'education.num', 'capital.gain', 'capital.loss', 'hours.per.week']
num_features = [preprocess_numerical(X_train, X_cv, X_test, col) for col in num_columns]

# Combine all features
all_features = cat_features + [sex_features] + num_features
X_train_preprocessed = hstack([feature[0] for feature in all_features])
X_cv_preprocessed = hstack([feature[1] for feature in all_features])
X_test_preprocessed = hstack([feature[2] for feature in all_features])

#In[]
X_train_preprocessed

#In[]
############XG Boost classifier###################
print("XGB Classifier")
from xgboost import XGBClassifier
weight = 6016/18984
max_dept = [3]
n_estimators = [200]
# et = [0.2,0.5,0.7,0.9]
# for e in [0.29,0.3,0.31]:
#     for depth in max_dept:
model = XGBClassifier(scale_pos_weight=weight,max_depth = 3, n_estimators=200,eta=0.3)
model.fit(X_train_preprocessed, Y_train)

yhat = model.predict_proba(X_train_preprocessed)
final_pred = []
for i in yhat:
    if i[0]>i[1]:
        final_pred.append(1-i[0])
    else:
        final_pred.append(i[1])
mse = 0
for i in range(len(Y_train)):
    mse += (list(Y_train)[i]-final_pred[i])**2
print("train mse",mse/len(Y_train))

#In[]
# evaluate model
yhat = model.predict_proba(X_cv_preprocessed)
# pred_test
final_pred = []
for i in yhat:
    if i[0]>i[1]:
        final_pred.append(1-i[0])
    else:
        final_pred.append(i[1])
mse = 0
for i in range(len(Y_cv)):
    mse += (list(Y_cv)[i]-final_pred[i])**2
print("cv mse",mse/len(Y_cv))

#In[]

model = XGBClassifier(scale_pos_weight=1-weight,max_depth = 3, n_estimators=200)
model.fit(X_train_preprocessed, Y_train)

yhat = model.predict_proba(X_test_preprocessed)

# pred_test
final_pred = []
for i in yhat:
    if i[0]>i[1]:
        final_pred.append(1-i[0])
    else:
        final_pred.append(i[1])

result = pd.DataFrame(columns=['ID','Prediction'])
result['ID'] = ID
result['Prediction'] = final_pred

result.to_csv("prediction_xgbclassifer_1.csv",index=False)

#In[]:
###################Linear Regression#####################
lr = LinearRegression()
lr.fit(X_train_preprocessed,Y_train)
train_pred = lr.predict(X_train_preprocessed)
cv_pred = lr.predict(X_cv_preprocessed)

mse = 0
print("Linear Regression")
for i in range(len(Y_train)):
    mse += (list(Y_train)[i]-train_pred[i])**2
print("train mse",mse/len(Y_cv))

#In[]
mse = 0
for i in range(len(Y_cv)):
    mse += (list(Y_cv)[i]-cv_pred[i])**2
print("cv mse",mse/len(Y_cv))

#In[]
pred_test = lr.predict(X_test_preprocessed)
result = pd.DataFrame(columns=['ID','Prediction'])
result['ID'] = ID
result['Prediction'] = pred_test
result.to_csv("prediction_linear_regression.csv",index=False) 
# %%
