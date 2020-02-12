#!/usr/bin/env python
# coding: utf-8

# In[68]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics


# In[69]:


# Data Handling: Load CSV
df = pd.read_csv("D:DS_TriS/creditcard.csv")


# In[70]:


# get to know list of features, data shape, stat. description.
print(df.shape)


# In[71]:


print("First 5 lines:")
print(df.head(5))


# In[72]:


print("describe: ")
print(df.describe())


# In[73]:


print("info: ")
print(df.info())


# In[74]:


"""Since all variables are of float and int type, so this data is easy to handle for modeling"""


# In[75]:


#Class_as_Pie_Chart


# In[76]:


fig, ax = plt.subplots(1,1)
ax.pie(df.Class.value_counts(),autopct="%1.1f%%", labels=['Genuine','Fraud'],colors=['green','red'])
plt.axis('equal')
plt.ylabel('')
plt.show()


# In[77]:


#plot Time to see if there is any trend
print("Time variable")
df["Time_Hr"] = df["Time"]/3600                   # convert to hours
print(df["Time_Hr"].tail(5))


# In[78]:


fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, figsize=(6,3))
ax1.hist(df.Time_Hr[df.Class==0],bins=48,color='g',alpha=0.5)
ax1.set_title('Genuine')
ax2.hist(df.Time_Hr[df.Class==1],bins=48,color='r',alpha=0.5)
ax2.set_title('Fraud')
plt.xlabel('Time (hrs)')
plt.ylabel('# transactions')


# In[79]:


"""number of transactions have almost similar dependence on time of the day for both the classes, I will drop "Time" but keep "Time_Hr"."""


# In[80]:


df = df.drop(['Time'],axis=1)


# In[81]:


#let us check another feature Amount
fig, (ax3,ax4) = plt.subplots(2,1, figsize = (6,3), sharex = True)
ax3.hist(df.Amount[df.Class==0],bins=50,color='g',alpha=0.5)
ax3.set_yscale('log') # to see the tails
ax3.set_title('Genuine') # to see the tails
ax3.set_ylabel('# transactions')
ax4.hist(df.Amount[df.Class==1],bins=50,color='r',alpha=0.5)
ax4.set_yscale('log') # to see the tails
ax4.set_title('Fraud') # to see the tails
ax4.set_xlabel('Amount ($)')
ax4.set_ylabel('# transactions')


# In[82]:


#interesting to note "all transaction amounts > 10K in Genuine Class only". 
#Also this amount feature is not on same scale as principle components. 
#So, I'll standardize the values of the 'Amount' feature using StandardScalar and save in data-frame for later use.


# In[83]:


from sklearn.preprocessing import StandardScaler
df['Scaled_amount'] = StandardScaler().fit_transform(df['Amount'].values.reshape(-1,1))
df = df.drop(['Amount'],axis = 1)


# In[84]:


df.head()


# In[85]:


#let us check correlations and shapes of those 25 principal components.
# Features V1, V2, ... V28 are the principal components obtained with PCA.
import seaborn as sns
import matplotlib.gridspec as gridspec
gs = gridspec.GridSpec(28, 1)
plt.figure(figsize=(6,28*4))
for i, col in enumerate(df[df.iloc[:,0:28].columns]):
    ax5 = plt.subplot(gs[i])
    sns.distplot(df[col][df.Class == 1], bins=50, color='r')
    sns.distplot(df[col][df.Class == 0], bins=50, color='g')
    ax5.set_xlabel('')
    ax5.set_title('feature: ' + str(col))
plt.show()


# In[86]:


#For some of the features, both the classes have similar distribution. 
#So, I don't expect them to contribute towards classifying power of the model. 
#So, it's best to drop them and reduce the model complexity, and hence the chances of overfitting. 


# In[87]:


#Now, it's time to split the data in test set (20%) and training set (80%). I'll define a function for it.


# In[88]:


def split_data(df, drop_list):             #drop_list has all the unneceesary columns
    df = df.drop(drop_list,axis=1)
    print(df.columns)
    #test train split time
    from sklearn.model_selection import train_test_split
    y = df['Class'].values               #target
    X = df.drop(['Class'],axis=1).values         #features
    print(X[0:5],y[0:5])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,
                                                    random_state=42, stratify=y)

    print("train-set size: ", len(y_train),
      "\ntest-set size: ", len(y_test))
    print("fraud cases in test-set: ", sum(y_test))
    return X_train, X_test, y_train, y_test


# In[89]:


#Below is funtion to define classifier and get predictions


# In[90]:


def get_predictions(clf, X_train, y_train, X_test):
    # create classifier
    clf = clf
    
    # fit it to training data
    clf.fit(X_train,y_train)
    
    # predict using test data
    y_pred = clf.predict(X_test)
    
    # Compute predicted probabilities: y_pred_prob
    y_pred_prob = clf.predict_proba(X_test)
    
    #for fun: train-set predictions
    train_pred = clf.predict(X_train)
    print('train-set confusion matrix:\n', metrics.confusion_matrix(y_train,train_pred)) 
    return y_pred, y_pred_prob


# In[91]:


#Function to print the classifier's scores
def print_scores(y_test,y_pred,y_pred_prob):
    print('test-set confusion matrix:\n', metrics.confusion_matrix(y_test,y_pred)) 
    print("recall score: ", metrics.recall_score(y_test,y_pred))
    print("precision score: ", metrics.precision_score(y_test,y_pred))
    print("f1 score: ", metrics.f1_score(y_test,y_pred))
    print("accuracy score: ", metrics.accuracy_score(y_test,y_pred))
    print("ROC AUC: {}".format(metrics.roc_auc_score(y_test, y_pred_prob[:,1])))


# In[92]:


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression


# In[93]:


# Case-NB-1 : do not drop anything

drop_list = []
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)


# In[94]:


# Case-NB-2 : drop some of principle components that have similar distributions in above plots

drop_list = ['V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)


# In[95]:


#'Clearly, by removing some of the reduntant principle components, I gain in model sensitivity and precision.'


# In[96]:


# Case-NB-3 : drop some of principle components + Time

drop_list = ['Time_Hr','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)


# In[97]:


#'As we can see by comapring Case-NB-3 scores with Case-NB-2 scores, "Time_Hr" is not helping much in classification. So, I can remove it safely.'


# In[98]:


# Case-NB-4 : drop some of principle components + Time + 'scaled_Amount'

drop_list = ['Scaled_amount','Time_Hr','V28','V27','V26','V25','V24','V23','V22','V20','V15','V13','V8']
X_train, X_test, y_train, y_test = split_data(df, drop_list)
y_pred, y_pred_prob = get_predictions(GaussianNB(), X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)


# In[99]:


"""I would say, Case-NB-4 gives me better model sensitivity (or recall) and precision as compared to Case-NB-1. So dropping some of redundant feature will ofcourse helps to make calculations fast and gain senstivity."""


# In[100]:


df = df.drop(drop_list,axis=1)
print(df.columns)


# In[101]:


# let us check recall score for logistic regression
# Case-LR-1
y_pred, y_pred_prob = get_predictions(LogisticRegression(C = 0.01, penalty = 'l1')
                                      , X_train, y_train, X_test)
print_scores(y_test,y_pred,y_pred_prob)


# In[102]:


"""As we see, by learning from full imbalanced dataset this default logistic regressor performs very poorly. So let us try to train it in tradional way i.e. from under-sampled data. So, take only that percent of genuine-class cases which is equal to all fraud-classes i.e. consider 50/50 ratio of both classes."""


# In[103]:


# get indices for fraud and genuine classes 
fraud_ind = np.array(df[df.Class == 1].index)
gen_ind = df[df.Class == 0].index
n_fraud = len(df[df.Class == 1])

# random selection from genuine class
random_gen_ind = np.random.choice(gen_ind, n_fraud, replace = False)
random_gen_ind = np.array(random_gen_ind)

# merge two class indices: random genuine + original fraud
under_sample_ind = np.concatenate([fraud_ind,random_gen_ind])

# Under sample dataset
undersample_df = df.iloc[under_sample_ind,:]
y_undersample  = undersample_df['Class'].values #target
X_undersample = undersample_df.drop(['Class'],axis=1).values #features

print("# transactions in undersampled data: ", len(undersample_df))
print("% genuine transactions: ",len(undersample_df[undersample_df.Class == 0])/len(undersample_df))
print("% fraud transactions: ", sum(y_undersample)/len(undersample_df))


# In[104]:


# let us train logistic regression with undersamples data
# Case-LR-2
# split undersampled data into 80/20 train-test datasets. 
# - Train model from this 80% fraction of undersampled data, get predictions from left over i.e. 20%.
drop_list = []
X_und_train, X_und_test, y_und_train, y_und_test = split_data(undersample_df, drop_list)
y_und_pred, y_und_pred_prob = get_predictions(LogisticRegression(C = 0.01, penalty = 'l1'), X_und_train, y_und_train, X_und_test)
print_scores(y_und_test,y_und_pred,y_und_pred_prob)


# In[105]:


"""As per expectations, wonderfull performance for completely balanced classes."""


# In[106]:


#Now, let us check its performance for the full skewed dataset. 
#Just to mention: "train" from undersampled data, and "test" on full data.


# In[107]:


# Case-LR-3
# "train" with undersamples, "test" with full data
# call classifier
lr = LogisticRegression(C = 0.01, penalty = 'l1')

# fit it to complete undersampled data
lr.fit(X_undersample, y_undersample)

# predict on full data
y_full = df['Class'].values #target
X_full = df.drop(['Class'],axis=1).values #features
y_full_pred = lr.predict(X_full)

# Compute predicted probabilities: y_pred_prob
y_full_pred_prob = lr.predict_proba(X_full)
print("scores for Full set")   
print('test-set confusion matrix:\n', metrics.confusion_matrix(y_full,y_full_pred)) 
print("recall score: ", metrics.recall_score(y_full,y_full_pred))
print("precision score: ", metrics.precision_score(y_full,y_full_pred))


# In[108]:


#I just want to compare the scores from GaussianNB with logistic-regression.
#get predictions for test-dataset (20% of full dataset) from both models.
#Aim is to compare recall score of Case-NB-4 with Case-LR-4.


# In[109]:


# Case-LR-4
y_p20_pred = lr.predict(X_test)
y_p20_pred_prob = lr.predict_proba(X_test)
print("scores for test (20% of full) set")   
print('test-set confusion matrix:\n', metrics.confusion_matrix(y_test,y_p20_pred)) 
print("recall score: ", metrics.recall_score(y_test,y_p20_pred))
print("precision score: ", metrics.precision_score(y_test,y_p20_pred))


# In[110]:


#So, now I have NB vs LR recall score of 0.878 vs 0.929.
#NB confusion matrix:
#[[55954 910] [ 12 86]]
#LR confusion matrix:
#[[53547 3317] [ 7 91]]
#Conclusions: Nodoubt, LR gives better model sensitivity,
#but positive predictive value for NB is more than double (although low for both). 
#As said in introduction, Naive-Bayes is just simple prob. calculator, no coeff. optimization by fitting etc. , 
#so this is quick learner. We can hopefully improve NB's performance by playing around with default threshold 
#on calculated probabilities, but still 89% of fraud cases are detected, bravo NB!

