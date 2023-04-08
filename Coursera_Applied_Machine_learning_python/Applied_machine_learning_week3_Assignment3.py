#!/usr/bin/env python
# coding: utf-8

# ---
# 
# _You are currently looking at **version 0.1** of this notebook. To download notebooks and datafiles, as well as get help on Jupyter notebooks in the Coursera platform, visit the Jupyter Notebook FAQ course resource._
# 
# ---

# In[1]:


import numpy as np
import pandas as pd


# ### Question 1
# Import the data from `assets/fraud_data.csv`. What percentage of the observations in the dataset are instances of fraud?
# 
# *This function should return a float between 0 and 1.* 

# In[2]:


def answer_one():
    # YOUR CODE HERE
    df = pd.read_csv('assets/fraud_data.csv')
    fraud_percentage = (df['Class'].sum()/len(df))
    return fraud_percentage
    raise NotImplementedError()


# In[ ]:





# In[3]:


# Use X_train, X_test, y_train, y_test for all of the following questions
from sklearn.model_selection import train_test_split

df = pd.read_csv('assets/fraud_data.csv')

X = df.iloc[:,:-1]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


# ### Question 2
# 
# Using `X_train`, `X_test`, `y_train`, and `y_test` (as defined above), train a dummy classifier that classifies everything as the majority class of the training data. What is the accuracy of this classifier? What is the recall?
# 
# *This function should a return a tuple with two floats, i.e. `(accuracy score, recall score)`.*

# In[4]:


def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    
    DC = DummyClassifier().fit(X_train,y_train)
    y_pred = DC.predict(X_test)
    
    accuracyscore = DC.score(X_test, y_test)
    recallscore = recall_score(y_test, y_pred)
    # YOUR CODE HERE
    return (accuracyscore, recallscore)
    raise NotImplementedError()


# In[ ]:





# ### Question 3
# 
# Using X_train, X_test, y_train, y_test (as defined above), train a SVC classifer using the default parameters. What is the accuracy, recall, and precision of this classifier?
# 
# *This function should a return a tuple with three floats, i.e. `(accuracy score, recall score, precision score)`.*

# In[5]:


def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC
    clf = SVC().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracyscore = clf.score(X_test, y_test)
    recallscore = recall_score(y_test, y_pred)
    precisionscore = precision_score(y_test, y_pred)
    return (accuracyscore, recallscore, precisionscore)
    # YOUR CODE HERE
    raise NotImplementedError()


# In[ ]:





# ### Question 4
# 
# Using the SVC classifier with parameters `{'C': 1e9, 'gamma': 1e-07}`, what is the confusion matrix when using a threshold of -220 on the decision function. Use X_test and y_test.
# 
# *This function should return a confusion matrix, a 2x2 numpy array with 4 integers.*

# In[14]:


def answer_four():
    from sklearn.metrics import confusion_matrix
    from sklearn.svm import SVC
    svm = SVC(C = 1e9, gamma = 1e-07)
    svm.fit(X_train, y_train)
    svm_predicted = svm.decision_function(X_test)>-220
    confusion = confusion_matrix(y_test, svm_predicted)
    return confusion
    # YOUR CODE HERE
    raise NotImplementedError()


# In[ ]:





# ### Question 5
# 
# Train a logisitic regression classifier with default parameters using X_train and y_train.
# 
# For the logisitic regression classifier, create a precision recall curve and a roc curve using y_test and the probability estimates for X_test (probability it is fraud).
# 
# Looking at the precision recall curve, what is the recall when the precision is `0.75`?
# 
# Looking at the roc curve, what is the true positive rate when the false positive rate is `0.16`?
# 
# *This function should return a tuple with two floats, i.e. `(recall, true positive rate)`.*

# In[13]:


def draw_pr_curve():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import roc_curve, auc

    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    y_scores_lr = lr.decision_function(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)
    closest_zero = np.argmin(np.abs(thresholds))
    closest_zero_p = precision[closest_zero]
    closest_zero_r = recall[closest_zero]

    import matplotlib.pyplot as plt
    plt.figure()
    plt.xlim([0.0, 1.01])
    plt.ylim([0.0, 1.01])
    plt.plot(precision, recall, label='Precision-Recall Curve')
    plt.plot(closest_zero_p, closest_zero_r, 'o', markersize = 12, fillstyle = 'none', c='r', mew=3)
    plt.xlabel('Precision', fontsize=16)
    plt.ylabel('Recall', fontsize=16)
    plt.axes().set_aspect('equal')
    plt.show()

draw_pr_curve()


# In[12]:


def draw_roc_curve():
    get_ipython().run_line_magic('matplotlib', 'notebook')
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression()
    lr.fit(X_train, y_train)
    y_scores_lr = lr.decision_function(X_test)

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)
    roc_auc_lr = auc(fpr_lr, tpr_lr)

    plt.figure()
    plt.xlim([-0.01, 1.00])
    plt.ylim([-0.01, 1.01])
    plt.plot(fpr_lr, tpr_lr, lw=3, label='LogRegr ROC curve (area = {:0.2f})'.format(roc_auc_lr))
    plt.xlabel('False Positive Rate', fontsize=16)
    plt.ylabel('True Positive Rate', fontsize=16)
    plt.title('ROC curve (1-of-10 digits classifier)', fontsize=16)
    plt.legend(loc='lower right', fontsize=13)
    plt.plot([0, 1], [0, 1], color='navy', lw=3, linestyle='--')
    plt.axes().set_aspect('equal')
    plt.show()
    
draw_roc_curve()


# In[7]:


# YOUR CODE HERE
def answer_five():
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve
    lr = LogisticRegression().fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    precision, recall, thresholds = precision_recall_curve(y_test, lr.decision_function(X_test))
    plt.plot(precision, recall, label = "Precision-recall curve")
    plt.xlabel("Precision")
    plt.ylabel("Recall")
    precision_recall = np.column_stack((precision, recall))
    precision_recall_df = pd.DataFrame(precision_recall, columns=['Precision', 'Recall'])
    precision_recall_df = precision_recall_df[precision_recall_df['Precision'] >= 0.75]

    # find the recall when the precision is 0.75
    recall_at_075_precision = precision_recall_df['Recall'].max()
    
    # finding true positive rate when the false positive rate is 0.16
    idx = (np.abs(recall - 0.16)).argmin()
    return (recall_at_075_precision, precision[idx])
    raise NotImplementedError()
answer_five()


# In[ ]:





# ### Question 6
# 
# Perform a grid search over the parameters listed below for a Logisitic Regression classifier, using recall for scoring and the default 3-fold cross validation. (Suggest to use `solver='liblinear'`, more explanation [here](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html))
# 
# `'penalty': ['l1', 'l2']`
# 
# `'C':[0.01, 0.1, 1, 10]`
# 
# From `.cv_results_`, create an array of the mean test scores of each parameter combination. i.e.
# 
# |      	| `l1` 	| `l2` 	|
# |:----:	|----	|----	|
# | **`0.01`** 	|    ?	|   ? 	|
# | **`0.1`**  	|    ?	|   ? 	|
# | **`1`**    	|    ?	|   ? 	|
# | **`10`**   	|    ?	|   ? 	|
# 
# <br>
# 
# *This function should return a 4 by 2 numpy array with 8 floats.* 
# 
# *Note: do not return a DataFrame, just the values denoted by `?` in a numpy array.*

# In[14]:


def answer_six():    
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression
    
    clf = LogisticRegression(solver='liblinear')
    param_grid = {'C':[0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']}
    
    grid_search = GridSearchCV(clf, param_grid, scoring = 'recall')
    grid_search.fit(X_train, y_train)
    cv_result = grid_search.cv_results_
    mean_test_score = cv_result['mean_test_score']
    result = np.array(mean_test_score).reshape(4,2)
    # YOUR CODE HERE
    return result
    raise NotImplementedError()
answer_six()


# In[ ]:





# In[ ]:


# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):
    get_ipython().run_line_magic('matplotlib', 'notebook')
    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(5,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10])
    plt.yticks(rotation=0);

#GridSearch_Heatmap(answer_six())


# In[ ]:





# In[ ]:




