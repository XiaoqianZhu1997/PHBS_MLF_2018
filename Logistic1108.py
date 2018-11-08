# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 09:35:34 2018

@author: Xie
"""

# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import seaborn
import pickle
import itertools
from sklearn.cross_validation import KFold
from sklearn.linear_model import LogisticRegression

# In[2]:
# loading .data type files
f = open("C:\\Users\\姚宝马\\Desktop\\train_test.data", 'rb')
# load data
dataset = pickle.load(f)
#generating variables
X_test1=dataset[0]
X_test2=dataset[1]
X_train1=dataset[2]
X_train2=dataset[3]
y_test1=dataset[4]
y_test2=dataset[5]
y_train1=dataset[6]
y_train2=dataset[7]


# In[3]:
# define plot confusion matrix function

from sklearn.metrics import confusion_matrix,GridSearchCV,precision_recall_curve,auc,roc_auc_score,roc_curve,recall_score,classification_report 
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=0)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        #print("Normalized confusion matrix")
    else:
        1#print('Confusion matrix, without normalization')

    #print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
# In[4]:
# Logistic Regression
   
   
# Logistic regression best estimator
log_regl = LogisticRegression()
log_reg1_param = {'penalty' : ['l1','l2'], 'C' : [0.01, 0.1, 1, 10, 100]}
gridCV_log_regl = GridSearchCV(log_regl, log_reg1_param, scoring = 'recall', cv = 10, refit = True, verbose = 1, n_jobs = -1)
gridCV_log_regl.fit(X_train1, y_train1)
log_reg1_best_parameters = gridCV_log_regl.best_params_
print('The best parameters for using Logistic Regression is', log_reg1_best_parameters) 

# In[5]

##########   Finishing choosing parameters

# Use this C_parameter to build the final model with the whole training dataset and predict the classes in the test
lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train1,y_train1.values.ravel())
y_pred1 = lr.predict(X_test1.values)

#accuracy calculation
lr.score(X_test1, y_test1)
print('accuracy of training set1: {:.4f}'.format(lr.score(X_train1,y_train1)))
print('accuaracy of test set1: {:.4f}'.format(lr.score(X_test1, y_test1)))

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test1,y_pred1)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

# ROC CURVE

y_pred1_score = lr.fit(X_train1,y_train1.values.ravel()).decision_function(X_test1.values)

fpr, tpr, thresholds = roc_curve(y_test1.values.ravel(),y_pred1_score)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic_undersampling')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

##  Undersampling finished


lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train2,y_train2.values.ravel())
y_pred2 = lr.predict(X_test2.values)

lr.score(X_test2, y_test2)
print('accuracy of training set2: {:.4f}'.format(lr.score(X_train2,y_train2)))
print('accuaracy of test set2: {:.4f}'.format(lr.score(X_test2, y_test2)))
#accuracy calculation

# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test2,y_pred2)
np.set_printoptions(precision=2)

print("Recall metric in the testing dataset: ", cnf_matrix[1,1]/(cnf_matrix[1,0]+cnf_matrix[1,1]))

# Plot non-normalized confusion matrix
class_names = [0,1]
plt.figure()
plot_confusion_matrix(cnf_matrix
                      , classes=class_names
                      , title='Confusion matrix')
plt.show()

# ROC CURVE
y_pred2_score = lr.fit(X_train2,y_train2.values.ravel()).decision_function(X_test2.values)

fpr, tpr, thresholds = roc_curve(y_test2.values.ravel(),y_pred2_score)
roc_auc = auc(fpr,tpr)

# Plot ROC
plt.title('Receiver Operating Characteristic_Oversampling')
plt.plot(fpr, tpr, 'b',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.0])
plt.ylim([-0.1,1.01])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

# Precision Recall
from itertools import cycle

lr = LogisticRegression(C = 0.01, penalty = 'l1')
lr.fit(X_train1,y_train1.values.ravel())
y_pred_undersample_proba = lr.predict_proba(X_test1.values)

thresholds = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal', 'red', 'yellow', 'green', 'blue','black'])

plt.figure(figsize=(5,5))

j = 1
for i,color in zip(thresholds,colors):
    y_test_predictions_prob = y_pred_undersample_proba[:,1] > i
    
    precision, recall, thresholds = precision_recall_curve(y_test1,y_test_predictions_prob)
    
    # Plot Precision-Recall curve
    plt.plot(recall, precision, color=color,
                 label='Threshold: %s'%i)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall example')
    plt.legend(loc="lower left")