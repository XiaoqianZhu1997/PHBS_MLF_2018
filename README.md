# Proposal_Credit Card Fraud Detection
## 1. Team member:
1701213050 [Lei Jiayuan](https://github.com/JiayuanLei)\
1701213124 [Xie Ting](https://github.com/XieTing1995)\
1701213135 [Yao Huanchen](https://github.com/HuanchenYao)\
1701213173 [Zhu Xiaoqian](https://github.com/XiaoqianZhu1997)

## 2. Background:
It is important that credit card companies are able to recognize **fraudulent** credit card transactions so that customers are not charged for items that they did not purchase. Therefore, the project's goal is to **identify fraudulent credit card transactions**.

## 3. Dataset Description:
- The dataset is called [Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) obtained from Kaggle website.
  And here we provide some [samples of the data](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/data/creditcard_sample.csv).
- It contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. However, the dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions. Therefore, we will re-sample the dataset as to offset this imbalance with the hope of arriving at a more robust and fair decision boundary.

- Dependent Variables:
  Class: 1 for fraudulent transactionns; 0 otherwise
- **Independent Variables**:
  1) Time: Number of seconds elapsed between this transaction and the first transaction in the dataset.
  2) Amount: Transaction amount, which can be used for example-dependant cost-senstive learning.
  3) V1-V28: the principal components obtained with PCA, which help predict Fraud. They also have been scaled.

## 4. Data preprocessing:
- We put our Data preprocessing details under the file [PHBS_MLF_2018/data_preprocessing.ipynb](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/data_preprocessing.ipynb)
- In this step, we tried undersampling method and oversampling method
- For each method, we tried different machine learning method to classify the data.
- **Further research direction**: 
  1) To deal with the outliers of each feature and see their effects on the accuracy rate; 
  2) To try oversample method and compare the new outcome with that of under sample method.  
  
  ![the wrong way](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/imbalanced_data_KFold1.png)
  
  ## **Remark**:
In the above processes, we under(over)sampled before doing cross-validation, which means we use the information of whole data, and the spliting into validation set & train set is after using these infos. However, the information from validation set should not used at taht time. So we need to do under(over)smpling during the validation.
It makes no sense to create instances based on our current minority class and then exclude an instance for validation, pretending we didn’t generate it using data that is still in the training set.

  
  ![the right way](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/imbalanced_data_KFold2.png)

## 5. Model:
Method: We mainly try **LogisticRegression/KNN/SVM/DecisionTree/RandomForest** to choose a better and appropirate model.
**Logistic Regression**
Under this method, we first use grid search to find the best (hyper)parameter. We find that {C=0.01, penalty=l1} is the best parameter. Therefore, we use this parameter to define the confusion matrix and plot the precision-recall curve.
![the confusion matrix](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/LR.png)

![the precision-recall curve](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/LR_prec.png)

**KNN**
Under this method, we first determine the range of KNN's k. As 1-NN always gives 100% train accuracy, so we excluded it out. And this is a binary case, we thought it's better to not use an even number like 2, then we exluded 2-NN out too. Therefore, the range starts from 3. We find that the best parameters for using KNN is {'algorithm': 'auto', 'n_neighbors': 4}. 
We get the confusion matrix and precision-recall curve as follows. 
![the confusion matrix](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/KNN.png)

![the precision-recall curve](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/KNN_prec.png)

**SVM**
Under this method, we find that the best parameters for using Surpport Vector Classifier is {'C': 3, 'kernel': 'rbf'}.
However, when defining confusion matrix, we have met a problem.

**Decision Tree**
By using grid search, we find the best parameter for Decision tree Classifier is {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 6}. 
We get the confusion matrix and precision-recall curve as follows.
![the confusion matrix](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/DT.png)

![the precision-recall curve](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/DT_prec.png)


**Random Forest**
Under this method, The best parameter is {'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 10, 'n_estimators': 100}.
We get the confusion matrix and precision-recall curve as follows.
![the confusion matrix](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/RF.png)

![the precision-recall curve](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/RF_prec.png)


