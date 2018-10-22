# Proposal_Credit Card Fraud Detection
## 1. Team member:
1701213050 Lei Jiayuan\
1701213124 Xie Ting\
1701213135 Yao Huanchen\
1701213173 Zhu Xiaoqian

## 2. Background:
It is important that credit card companies are able to recognize fraudulent credit card transactions so that customers are not charged for items that they did not purchase. Therefore, the project's goal is to identify fraudulent credit card transactions.

## 3. Data set description:
- The dataset is called "Credit Card Fraud Detection", obtained from Kaggle website (https://www.kaggle.com/mlg-ulb/creditcardfraud).
- It contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. However, the dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. Therefore, we will re-sample the dataset as to offset this imbalance with the hope of arriving at a more robust and fair decision boundary.

- Dependent Variables:
  Class: 1 for fraudulent transactionns; 0 otherwise
- Independent Variables:
  1)Time: Number of seconds elapsed between this transaction and the first transaction in the dataset.
  2)Amount: Transaction amount, which can be used for example-dependant cost-senstive learning.
  3)V1-V28: the principal components obtained with PCA, which help predict Fraud. They also have been scaled.

## 4. Data preprocessing:
- We put our Data preprocessing details under [PHBS_MLF_2018/data_preprocessing.ipynb]
- Further research direction: 1) To deal with the outliers of each feature and see their effects on the accuracy rate; 2) To try oversample method and compare the new outcome with that of under sample method.  

## 5. Model:
Method: Try KNN/logistic/SVM/RandomForest/bagging/AdaBoost to choose a better model.
