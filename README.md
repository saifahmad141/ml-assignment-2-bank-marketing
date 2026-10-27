Machine Learning Assignment 2: Bank Marketing Classification

1. Problem Statement

The objective of this project is to build and compare five machine learning classification models to predict whether a customer will subscribe to a term deposit based on the Bank Marketing dataset.

The project evaluates the performance of five classification algorithms using multiple evaluation metrics and identifies the most suitable model based on the results.

2. Dataset Description

Dataset: Bank Marketing Dataset
File: bank.csv

The dataset contains information about customers and marketing campaign interactions.

* Total instances: 4,521
* Total columns: 17
* Input features: 16
* Target variable: y
* Target classes: yes and no

The target variable indicates whether a customer subscribed to a term deposit.

Target Class Distribution

Class	Number of Records	Percentage
No	4,000	88.48%
Yes	521	11.52%
Total	4,521	100%

The dataset has an imbalanced target distribution, with substantially more customers in the no class than in the yes class.

3. Data Preprocessing and Experimental Setup

The following steps were performed:

1. Loaded and inspected the dataset.
2. Checked for missing values and duplicate records.
3. Separated the input features from the target variable.
4. Identified numerical and categorical features.
5. Applied appropriate preprocessing to numerical and categorical data through machine learning pipelines.
6. Encoded the target variable as a binary class.
7. Split the data into training and testing sets.

Train-Test Split

Dataset	Number of Records
Training set	3,616
Testing set	905
Total	4,521

The five classifiers were evaluated using the same train-test split.

4. Machine Learning Models

The following five classification algorithms were implemented:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

5. Model Performance Comparison

The models were evaluated using Accuracy, Area Under the ROC Curve (AUC), Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

Model	Accuracy	AUC	Precision	Recall	F1 Score	MCC
Logistic Regression	0.8928	0.8913	0.5636	0.2981	0.3899	0.3579
Decision Tree	0.8586	0.6942	0.4032	0.4808	0.4386	0.3602
KNN	0.8840	0.7692	0.4872	0.1827	0.2657	0.2477
Gaussian Naive Bayes	0.8210	0.7881	0.3092	0.4519	0.3672	0.2737
Support Vector Machine	0.8939	0.8830	0.5909	0.2500	0.3514	0.3374

6. Observations

* Accuracy: Support Vector Machine achieved the highest accuracy of 0.8939, closely followed by Logistic Regression at 0.8928.
* AUC: Logistic Regression achieved the highest AUC of 0.8913, indicating the strongest overall ranking performance among the evaluated models according to this metric.
* Precision: Support Vector Machine achieved the highest precision of 0.5909.
* Recall: Decision Tree achieved the highest recall of 0.4808, making it the strongest model for identifying positive subscription cases at the selected classification threshold.
* F1 Score: Decision Tree achieved the highest F1 Score of 0.4386, reflecting the best balance between precision and recall among these results.
* MCC: Decision Tree achieved the highest MCC of 0.3602, narrowly exceeding Logistic Regression at 0.3579.

The dataset is imbalanced, so accuracy alone does not provide a complete picture of model performance. Precision, recall, F1 Score, AUC, and MCC are also important when comparing these classifiers.

7. Overall Conclusion

Support Vector Machine achieved the highest accuracy, while Logistic Regression achieved the highest AUC. Decision Tree achieved the highest recall, F1 Score, and MCC.

Considering the imbalanced target distribution and the goal of identifying customers who subscribe to term deposits, the choice of the best model depends on the evaluation metric and the practical objective.

Based on the reported results, Decision Tree provides the strongest balance of precision and recall according to the F1 Score and achieves the highest MCC, while Logistic Regression offers the best AUC.

8. Repository Contents

The repository includes:

* bank.csv — original dataset
* test_data.csv — test dataset used for model evaluation
* ML_Assignment2_Bank_Marketing_Cleaned.ipynb — notebook containing the implementation and evaluation
* decision_tree.joblib — saved Decision Tree model
* knn.joblib — saved KNN model
* logistic_regression.joblib — saved Logistic Regression model
* naive_bayes.joblib — saved Gaussian Naive Bayes model
* support_vector_machine.joblib — saved SVM model

9. Streamlit Application

A Streamlit application will be used to demonstrate the trained models. The intended interface will allow users to upload a CSV file, select a saved classifier, and view supported evaluation metrics and classification results.

Live application: To be added after deployment.

10. GitHub Repository

Repository: https://github.com/saifahmad141/ml-assignment-2-bank-marketing
