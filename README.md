# telecom_churn_prediction

## Overview
The telecom operator Interconnect would like to be able to forecast their churn of clients. If it's discovered that a user is planning to leave, they will be offered promotional codes and special plan options. Interconnect's marketing team has collected some of their clientele's personal data, including information about their plans and contracts.

## Summary
In this project I managed to get roc_auc score of more then 0.9 on the test set. This means that the area under the curve of true positive rave vs. false positive rate when changing the threshold is large and far larger then chance (0.5). The best preforming model is Cat Boost Classifier that I tunned with 5 fold of cross validation and introduced different hyperparameters randomly with Randomized Search CV. The class are not balanced so I added synthetic observation to over sample the minority class using SMOTE in a pipeline. Akey step to achieve a good modeling was to understand the trend and seasonality of the data from a time series analysis and angineer new features to represent it.  

