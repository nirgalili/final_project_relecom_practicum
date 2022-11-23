# telecom_churn_prediction

## Overview
The telecom operator Interconnect would like to be able to forecast their churn of clients. If it's discovered that a user is planning to leave, they will be offered promotional codes and special plan options. Interconnect's marketing team has collected some of their clientele's personal data, including information about their plans and contracts.

## Summary
* **Preparing data for supervised learning**:
    * engineered a few extra features which might be helpful
        * `months_registered` as the ratio of total charges to monthly charges, representing how long users were registered
        * `month` and `year` as features that might show seasonality in regards to churning
    * converted all Yes/No categorical features to 0/1 int types
    * converted `multiple lines` to a 0/1/2 value `num_lines` representing how many phone lines the user has
    * dropped unneeded columns (`customer_id`, `end_date`, `multiple_lines` and `begin_date`)
    * performed dummy encoding / OHE for the `gender`, `type`, `payment_method` and `internet_service` features
    * divided the data 4:1 between train/validation and test sets
    * prepared a pre-processing step for the modeling pipelines that scales all numerical features using StandardScaler() and produced synthetic observation to the minority positive class using SMOTE().

* **Modeling and hyperparameter tuning**:
    * created a helper function to perform the repetitive parts of hyperparameter tuning
    * tuned parameters for 3 different functions using randomsearch cross validation via 5 StratifiedKfold folds:
        * LogisticRegression: tuned the solver, penalty and  Inverse of regularization strength parameter.
        * RandomForestClassifier: tuned the max_features and n_estimators parameters
        * LGBMClassifier: tuned the boosting_type, learning_rate and n_estimators parameters
    * the LGBMClassifier model had the highest score with 0.96 maximum AUC-ROC score.

* **Testing the models**:
    * used the above optimal parameters train the models on the full training set
    * as a sanity check, added a DummyClassifier() that always predicts the most common class
    * noticed that in terms of learning and prediction time, RF > LGBM > LR, but all models took less than 1 second
    * achieved AUC-ROC scores of 0.99 for the LGBM model, with the next best model being RF and last LR
    * achieved accuracy scores of \~0.87 for RF, \~0.94 for LR and \~0.96 for LGBM

