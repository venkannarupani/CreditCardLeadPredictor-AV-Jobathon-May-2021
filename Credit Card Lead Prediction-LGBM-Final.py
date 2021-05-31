#!/usr/bin/env python
# coding: utf-8

# ## Credit Card Lead Prediction - Problem Statement

# Happy Customer Bank is a mid-sized private bank that deals in all kinds of banking products, like Savings accounts, Current accounts, investment products, credit products, among other offerings.
# 
# The bank also cross-sells products to its existing customers and to do so they use different kinds of communication like tele-calling, e-mails, recommendations on net banking, mobile banking, etc. 
# 
# In this case, the Happy Customer Bank wants to cross sell its credit cards to its existing customers. The bank has identified a set of customers that are eligible for taking these credit cards.
# 
# Now, the bank is looking for your help in identifying customers that could show higher intent towards a recommended credit card, given:
# 
# 1. Customer details (gender, age, region etc.)
# 
# 2. Details of his/her relationship with the bank (Channel_Code,Vintage, 'Avg_Asset_Value etc.)

# ## Table of Content

# * __Step 1: Importing Required Libraries__
#     
# * __Step 2: Data Loading & Inspection__
#     
# * __Step 3: Data Wrangling__
#     
# * __Step 4: Exploratory Data Analysis (EDA)__
#     
# * __Step 5: Building Model__
#     
# * __Step 6: Predicting on Test Data and Generating submission file__

# ### Step 1: Importing Required Libraries

# In[1]:


import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')


# ### Step 2: Data Loading & Inspection

# In[2]:


# Loading Train and Test Datasets
train = pd.read_csv("train_s3TEQDk.csv")
test = pd.read_csv("test_mSzZ8RL.csv")


# In[3]:


# Viewing the shape of the train and test datasets
train.shape,test.shape


# * __We have 2,45,725 rows and 11 columns in Train set whereas, Test set has 1,05,312 rows and 10 columns.__

# In[4]:


# Viewing the Train dataset
train.head()


# In[5]:


# Viewing the Test dataset
test.head()


# In[6]:


# Viewing the datatypes of Train dataset
train.dtypes


# In[7]:


# Viewing the datatypes of Test dataset
test.dtypes


# In[8]:


# Finding the Ratio of null values in Train dataset
train.isnull().sum()/train.shape[0] *100


# In[9]:


# Finding the Ratio of null values in Test dataset
test.isnull().sum()/test.shape[0] *100


# * __We have 11.93% and 11.89% of missing values in Credit_Product column of Train data and Test data respectively.__

# In[10]:


# Finding the number of categorical features in Train dataset
categorical = train.select_dtypes(include =[np.object])
print("Categorical Features in Train Set:",categorical.shape[1])

# Finding the number of numerical features in Train dataset
numerical= train.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features in Train Set:",numerical.shape[1])


# In[11]:


# Finding the number of categorical features in Test dataset
categorical = test.select_dtypes(include =[np.object])
print("Categorical Features in Test Set:",categorical.shape[1])

# Finding the number of numerical features in Test dataset
numerical= test.select_dtypes(include =[np.float64,np.int64])
print("Numerical Features in Test Set:",numerical.shape[1])


# ### Step 3: Data Wrangling

# Missing Values:
# 
# The unwanted presence of missing values in the training data often reduces the accuracy of a model or leads to a biased model. It leads to inaccurate predictions. This is because we don’t analyse the behavior and relationship with other variables correctly. So, it is important to treat missing values well.
# 
# Missing values under 'Credit_Product' column have been put under seperate category during label encoding.

# In[12]:


# Finding out the column(s) in which null values are available in the Train dataset
train.isnull().sum()


# * __Credit_Product has 29,325 null values in the Train dataset.__

# In[13]:


# Finding out the column(s) in which null values are available in the Test dataset
test.isnull().sum()


# * __Credit_Product has 12,522 null values in the Test dataset.__

# In[14]:


# Viewing the distinct values in 'Credit_Product' column of Train dataset
train['Credit_Product'].value_counts()


# In[15]:


# Viewing the distinct values in 'Credit_Product' column of Train dataset through countplot
plt.figure(figsize=(8,5))
sns.countplot('Credit_Product',data=train)


# In[16]:


# Viewing the distinct values in 'Credit_Product' column of Test dataset
test['Credit_Product'].value_counts()


# In[17]:


# Viewing the distinct values in 'Credit_Product' column of Test dataset through countplot
plt.figure(figsize=(8,5))
sns.countplot('Credit_Product',data=test)


# In[18]:


# Viewing missing values in Train and Test datasets
train['Credit_Product'].isnull().sum(),test['Credit_Product'].isnull().sum()


# ### Step 4: Exploratory Data Analysis (EDA)

# In[19]:


# Viewing the Columns of Train dataset
train.columns


# In[20]:


# Viewing the first 5 records of Train dataset
train.head()


# In[21]:


# Viewing the distinct values in 'Gender' column of Train dataset
train['Gender'].value_counts()


# In[22]:


# Viewing the distinct values in 'Region_Code' column of Train dataset
train['Region_Code'].value_counts()


# In[23]:


# Viewing the distinct values in 'Occupation' column of Train dataset
train['Occupation'].value_counts()


# In[24]:


# Viewing the distinct values in 'Channel_Code' column of Train dataset
train['Channel_Code'].value_counts()


# In[25]:


# Viewing the distinct values in 'Vintage' column of Train dataset
train['Vintage'].value_counts()


# In[26]:


# Viewing the distinct values in 'Is_Active' column of Train dataset
train['Is_Active'].value_counts()


# __We see there are no irregularities in the data values of various columns.__

# In[27]:


# Viewing the distinct values in 'Gender' column of Train dataset through countplot
plt.figure(figsize=(8,5))
sns.countplot('Gender',data=train,palette='ocean')


# * __Male customers are more than Female customers.__

# In[28]:


# Viewing the distinct values in 'Occupation' column of Train dataset through countplot
plt.figure(figsize=(25,7))
sns.countplot('Occupation',data=train,palette='spring')


# * __Self-employed customers are more. Entrepreneurs are very less.__

# In[29]:


# Viewing the distinct values in 'Channel_Code' column of Train dataset through countplot
plt.figure(figsize=(8,5))
sns.countplot('Channel_Code',data=train,palette='summer')


# * __X1 channel code is highest. X4 is the least.__

# In[30]:


# Viewing the distinct values in 'Is_Active' column of Train dataset through countplot
plt.figure(figsize=(8,5))
sns.countplot('Is_Active',data=train,palette='autumn')


# * __There are more inactive customers than the active customers.__

# In[31]:


# Viewing the distinct values in label of 'Is_Lead' column of Train dataset through countplot
plt.figure(figsize=(8,5))
sns.countplot('Is_Lead',data=train,palette='autumn')


# * __There are more customers who do not produce lead for credit card.__

# In[32]:


# Importing Pandas Profiling Library
from pandas_profiling import ProfileReport

# Generating Pandas Profiling Report on Training data for Exploratory Data Analysis including finding out unique values
profreport = ProfileReport(train)
profreport.to_file(output_file='trainreport.html')


# In[33]:


# Viewing the Profile Report within the jupyter notebook
profreport.to_notebook_iframe()


# In[34]:


# Viewing the correlation among the variables of train dataset w.r.t. 'Is_Label'
train.corr()


# In[35]:


# Finding the correlation among various features

plt.figure(figsize = (12,12))
sns.heatmap(train.corr(), annot = True, cmap = "RdYlGn")

plt.show()


# ### Step 5: Building Model

# In[36]:


# Viewing the Train dataset
train.head()


# In[37]:


# Viewing the datatypes of Train dataset
train.dtypes


# In[38]:


# Chanding the type of 'Credit_Product'
train['Credit_Product']=train['Credit_Product'].astype(str)


# In[39]:


# Viewing the info of train dataset
train.info()


# In[40]:


# Label Encoding of Fields in Train dataset
le_train = LabelEncoder()
 
train['Gender']= le_train.fit_transform(train['Gender'])
train['Occupation']= le_train.fit_transform(train['Occupation'])
train['Credit_Product']= le_train.fit_transform(train['Credit_Product'])
train['Vintage']= le_train.fit_transform(train['Vintage'])
train['Region_Code']= le_train.fit_transform(train['Region_Code'])
train['Channel_Code']= le_train.fit_transform(train['Channel_Code'])
train['Age']= le_train.fit_transform(train['Age'])
train['Is_Lead']= le_train.fit_transform(train['Is_Lead'])
train['Is_Active']= le_train.fit_transform(train['Is_Active'])
train['ID']= le_train.fit_transform(train['ID'])


# * __Encoding the required columns from training dataset completed.__

# In[41]:


# Viewing the columns of Train dataset
train.columns


# In[42]:


# Viewing the Train dataset after label encoding
train.head()


# In[43]:


# Viewing the datatypes of Train dataset after label encoding
train.dtypes


# In[44]:


# Viewing if any missing values available in train dataset
train.isnull().sum()


# In[45]:


# Viewing the distinct values and their count of 'Credit_Product' in train dataset
train['Credit_Product'].value_counts()


# In[46]:


# Label Encoding of Fields in Test dataset
test['Credit_Product']=test['Credit_Product'].astype(str)

le_test = LabelEncoder()

test['Gender']= le_test.fit_transform(test['Gender'])
test['Occupation']= le_test.fit_transform(test['Occupation'])
test['Credit_Product']= le_test.fit_transform(test['Credit_Product'])
test['Vintage']= le_test.fit_transform(test['Vintage'])
test['Region_Code']= le_test.fit_transform(test['Region_Code'])
test['Channel_Code']= le_test.fit_transform(test['Channel_Code'])
test['Age']= le_test.fit_transform(test['Age'])
test['Is_Active']= le_test.fit_transform(test['Is_Active'])
test['ID']= le_test.fit_transform(test['ID'])
#test["Avg_Account_Balance"]=le_test.fit_transform(test["Avg_Account_Balance"])


# In[47]:


# Viewing the Test dataset after label encoding
test.head()


# In[48]:


# Viewing the datatypes of Test dataset after label encoding
test.dtypes


# #### Using Pycaret to find out the best classification model

# In[49]:


# Import classification module from Pycaret 
from pycaret.classification import * 


# In[50]:


# Setting up the environment for pycaret
reg1 = setup(train, target = 'Is_Lead', train_size = 0.8, session_id=156, log_experiment=True, experiment_name='CardLead')


# In[51]:


# Find out the best model based on AUC
best = compare_models(sort = 'AUC')


# * __Predictions using Light Gradient Boosting Machine are 86.09% accurate with highest AUC of 0.8735.__

# In[52]:


# Create Light Gradient Boosting Model
lightgbm = create_model('lightgbm')


# In[53]:


# Predict on lightgbm
pred_lightgbm = predict_model(lightgbm)


# In[54]:


# Finalize a lightgbm model
lightgbm_final = finalize_model(lightgbm)


# ### Step 6: Predicting on the Test Data and Generating submission file

# In[55]:


# Generate predictions on the Test dataset
test_pred = predict_model(lightgbm_final, data = test)


# In[56]:


# Viewing the predicted values ('Label' & 'Score') on the Test dataset
test_pred


# In[57]:


# Reading the submission file
submission = pd.read_csv('sample_submission_eyYijxG.csv')

# Assigning the predicted scores to 'Is_Lead' column of submission dataset
submission['Is_Lead'] = test_pred['Score']

# Writing the submission dataset output to the csv file
submission.to_csv('CreditCardLeadPredict_lightgbm_final.csv', index=False)


# ### End
