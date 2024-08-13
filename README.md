# Bank-Churn-Binary-Classification
Predicting whether or not a bank customer will churn (close their account) by using an XGBoost classifier

![image](https://github.com/user-attachments/assets/c5b8fec1-5ce5-44b4-8e5f-340f969eb541)

---

### Objective
This project was completed as part of Kaggle's Playground Series competition (Season 4, Episode 1). The main goal of this project is to employ binary classification techniques to predict whether or not a customer continues with their bank account or closes it, i.e. churns. More specifically, this project will be predicting the probability that a customer closes their account. In order to solve this binary classification problem, an XGBooster Classifier (XGB) will be used. Furthermore, the parameters of the XGB model will be optimized in order to obtain higher performance metrics. The dataset has been provided by Kaggle and is synthetic data generated from an actual Bank Customer Churn Prediction dataset. The target variable for the binary classification task is 'Exited', i.e. whether the customer has churned (1 = yes, 0 = no). The dataset also includes 12 predictor variables. 

---

### Methods 
Libraries Used
- pandas
- numpy
- matplotlib
- seaborn
- sklearn (LabelEncoder, train_test_split)
- sklearn.metrics (log_loss)
- xgboost

Data Preprocessing
- Converting object-type predictors to numerical w/ LabelEncoder()
- Removing unnecessary predictors ('id', 'CustomerId', 'Surname')

Exploratory Data Analysis (EDA)
- Summary statistics of the datasets (train and test)
- Bar chart viewing the distribution of the target variable
- Pie chart viewing the distribution of the target variable
- Histograms visualizing the distributions of the feature variables
- Heatmap showing potential correlations amongst feature variables

Building the Model
- Train test split (80% train, 20% test)
- XGB model
- Running trial to determine best parameters (eval_metric: 'mlogloss')
- Fitting the model with the optimal parameters

---

### General Results
The final XGB model yielded a training error of ~0.3176. Furthermore, the model yielded a competition score of 0.88537 (area under the ROC curve). 
