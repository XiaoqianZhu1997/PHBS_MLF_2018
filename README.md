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
- In this step, we tried undersampling method and oversampling method.
- After resampling, we use grid search cross-validation to find optimal (hyper)parameters and substitue those params into our prediction model.
The detailed (hyper)parameters that are treated as best parameters can be found in the [Third Part](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/ml_projec_v5.ipynb).
- **Further research direction**: 
  1) To deal with the outliers of each feature and see their effects on the accuracy rate; 
  2) To try oversample method and compare the new outcome with that of under sample method.  
  
**Here are something worth to mentioned:**
- 1. Firstly, the testing process should use the original data, test sets should be as representative as the "true" distribution as possible.
- 2. The second thing is that, until now, we were doing balancing the data before Grid search cross-validation to choose (hyper)parameters. However, this method can be problematic. Over(Under)-sampling before performing the K-fold split introudces identical samples in the training and test sets for each fold, which lead to an overestimate of model performance. And this [blog post](https://www.marcoaltini.com/blog/dealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation) gives a nice explanation. 
- Actually, over(under)-sample before cross-validation means we use the information of whole data, and the spliting into validation set & train set is after using these infos. However, the information from validation set should not used at taht time. So we need to do under(over)smpling during the validation.
- 3. Here are two figures we scratch from that blog post to visualize the problem:
  
  1. The wrong way:
  ![the wrong way](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/imbalanced_data_KFold1.png)

  2. The right way:
  ![the right way](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/imbalanced_data_KFold2.png)
 
 
 - So, we need to improve the way that we find the best parameters. In our notebook, we define a helper function to accomplish this thread, which can be found in the [Fourth Part](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/ml_projec_v5.ipynb).
 

## 5. Model:
Method: We mainly try **LogisticRegression/KNN/SVM/DecisionTree/RandomForest** to choose a better and appropirate model.

And in this part, we give the Condusion matrix and precision-recall curve figures of each models.

**The reason that we don't use ROC curve is that:**
- With imbalanced data, we've got a large number in Negative class, due to this large number, the increase of FPR is hard to observe. This will lead to (ROC curve) an over-optimistic estimation.
- In ROC curve, the x-axis is FPR = FP/N = FP/(FP+TN). In our case, num(N) >> num(P), a huge increase in FP can only bring small change in FPR. While the result is that a lot of Negative case was misclassified as Positive, but we cannont observe from ROC curve.

We can use an numerical example to illustrate this problem, suppose a dataset with Postive case (num=20) and Negative case(num=10000). At the begin, there are 20 Negative cases was misclassified.
  
Then FPR = 20/(20+9980) = 0.002, while if 20 Negative cases wa misclassified agian.
  FPR_2 = 40/(40+9960) = 0.004.

In ROC curve, this is just a tiny change. But the precision will change from 0.5 to 0.33, there will be a dramaticlly decrase in in the precision-recall curve.

### 5.1 **Logistic Regression**
Under this method, we first use grid search to find the best (hyper)parameter. We find that {'C': 0.01, 'penalty': 'l1'} is the best parameter. Therefore, we use this parameter to define the confusion matrix and plot the precision-recall curve.

  1. Confusion Matrix:
![](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/CM_LR.png)

  2. Precision-recall curve:
  
![](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/curve_LR.png)

### 5.2 **KNN**
Under this method, we first determine the range of KNN's k. As 1-NN always gives 100% train accuracy, so we excluded it out. And this is a binary case, we thought it's better to not use an even number like 2, then we exluded 2-NN out too. Therefore, the range starts from 3. We find that the best parameters for using KNN is {'algorithm': 'auto', 'n_neighbors': 7}. 
We get the confusion matrix and precision-recall curve as follows.

  1. Confusion Matrix:
![](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/CM_KNN.png)

  2. Precision-recall curve:
![](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/curve_KNN.png)

### 5.3 **SVM**
Under this method, we find that the best parameters for using Surpport Vector Classifier is {'C': 3, 'kernel': 'linear'}.

  1. Confusion Matrix:
![](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/CM_SVM.png)

  2. Precision-recall curve:
  
![](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/curve_SVM.png)

### 5.4 **Decision Tree**
By using grid search, we find the best parameter for Decision tree Classifier is {'criterion': 'entropy', 'max_depth': 6, 'min_samples_leaf': 7}. 
We get the confusion matrix and precision-recall curve as follows.

  1. Confusion Matrix:
![](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/CM_DT.png)

  2. Precision-recall curve:
  
![](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/curve_DT.png)


### 5.5 **Random Forest**
Under this method, The best parameter is {'max_features': 3, 'min_samples_leaf': 1, 'min_samples_split': 5, 'n_estimators': 200}.
We get the confusion matrix and precision-recall curve as follows.

  1. Confusion Matrix:
![](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/CM_RF.png)

  2. Precision-recall curve:
  
![](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/curve_RF.png)


## 6. Conclusion & Improvment:
### 6.1 Conclusions:
First, we look at the accuracy score, precision score, recall score betwwen different models:
![](https://github.com/XiaoqianZhu1997/PHBS_MLF_2018/blob/master/image/model_comparison.png)
1.  Accuracy score:
We won't do the comparsion based on this score, because this is a highly imblanced data, if one classifier predict that all the clients as non-fraudlent people, it will also give us a nice accuracy score. This is due to the relatively large number of the majority class compare to the minority class. So this is not a good metric in this case.

2. Precison score:
- We can find that in the above figure, precision score of all models are pretty low. Recall that the way to compute precision score is P = TP/(TP+FP), TP is linked to the minority class, the number is small, while in the denominator (TP+FP), FP is related to the majority class. So the precision score will always be quite small, it's hard to compare these tiny numbers. 

3. **Recall score:**
- Actually we will use this metric to do the comparison. Recall that R = TP/(TP+FN), from this definition we can get the reason that recall score looks better than precision score. And in this case, **the most important thing is that we detect all the client that will indeed be fraudulent**, compare to the accuracy, we concern more about **complete**.Recall score is the metric that do this thing exactly. While precision score is more focus on that in our prediction classes, how many cases are predicted accurately.

In the test set, the **Logistic Regression** gives us the highest value of recall score: 0.956. 


### 6.2 Improvments:
However, our method can be improved from these aspect:
1.  As we've mentioned in part 4, we need do the over(under)sampling during K-fold, and we did define a function to complete this method, but due to some heavy computation(even we set cv=3, it still took a long time to finish). 
2.  The way wo used to find the "best_parameters" is grid search, but we just list several pairs to ask the computer to choose between these lists. And we tried grid search repeatedly, we found that those results can be different in each operation. I was guessing that this fact is related to what we did just find the local optimal parasms rather than global optimal, so it will exist small difference in multiple trials.
3.  For the metric part, we can see that with different threshold, the precision-recall curve of KNN didn't change too much. 


