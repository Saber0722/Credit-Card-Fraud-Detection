# This is the main application file for model training and evaluation using streamlit

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


# Set page configuration
st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")

st.title("Credit Card Fraud Detection")
st.subheader("Logistic Regression Model Evaluation")
st.write("This application allows you to evaluate the performance of a logistic regression machine learning model on the given dataset to detect fraudulent transactions.")

# Display the raw dataset
st.subheader("Dataset")
st.write("The dataset contains credit card transactions with features and a target variable indicating whether the transaction is fraudulent or not.")
st.write("The dataset contains transactions made by credit cards in September 2013 by European cardholders.")
st.write("This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.")
st.write("It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.")
# Load the dataset
df= pd.read_csv("../data/creditcard.csv")
st.dataframe(df.head())

# Show countplot to portray class imbalance
st.subheader("Class Distribution")
st.write('From the following graph, we can see that the dataset is highly imbalanced.')
countplot= '../outputs/plots/class_distribution.png'
st.image(countplot, caption='Class Distribution', width=700)

# Correlation heatmap
st.subheader("Correlation Heatmap")
st.write("The correlation heatmap shows the correlation between different features in the dataset from which important ones have been picked to make analysis.")
correlation_heatmap = '../outputs/plots/correlation_matrix.png'
st.image(correlation_heatmap, caption='Correlation Heatmap', use_container_width=True)

# Positive and negative correlation of most impactful features
st.subheader("Positive and Negative Correlation of Most Impactful Features")
positive_correlation = '../outputs/plots/positive_correlation_distribution.png'
negative_correlation = '../outputs/plots/negative_correlation_distribution.png'
st.image(positive_correlation, caption='Positive Correlation', use_container_width=True)
st.image(negative_correlation, caption='Negative Correlation', use_container_width=True)
st.write("From the above graphs we can make the following observations:")
st.markdown("- Positive Correlation: The higher these values, the more likely the transaction is to be fraudulent.")
st.markdown("- Negative Correlation: The lower these values, the more likely the transaction is to be fraudulent.")

# Transaction amount distribution
st.subheader("Transaction Amount Distribution")
transaction_amount = '../outputs/plots/transaction_amount_distribution.png'
st.image(transaction_amount, caption='Transaction Amount Distribution', width=900)
st.write("The above graph shows the distribution of transaction amount in the dataset. The majority of transactions are below 2000.")
st.write("The distribution is right-skewed, indicating that most transactions are of lower amounts, while a few transactions are of higher amounts. This is common in credit card transaction datasets.")

# Time Distribution
st.subheader("Time Distribution")
st.write("The time distribution shows the distribution of transaction times in the dataset. The majority of transactions occur within a specific time range.")
time_distribution = '../outputs/plots/time_distribution.png'
st.image(time_distribution, caption='Time Distribution', width=900)

# Transaction Amount vs Time
st.subheader("Transaction Amount vs Time")
st.write("The scatter plot shows the relationship between transaction amount and time. It helps to identify any patterns or trends in the data.")
transaction_amount_time = '../outputs/plots/amount_vs_time.png'
st.image(transaction_amount_time, caption='Transaction Amount vs Time', width=900)
st.write("From the above graph, we can tell that there weren't any huge transactions performed by the user (fraudulent or not). Most of the transactions are below 2000 indicating that the fraudulent-user was being very wary of his transactions. The same can be said for the non-fraudulent user. The only difference is that the fraudulent user made a few transactions above 2000. This indicates that the fraudulent user was trying to hide his transactions by making small transactions and then making a few big ones to make it look like he was not trying to hide anything.")

# Pairplot of most impactful features
st.subheader("Pairplot of Most Impactful Features")
st.write("The pairplot shows the pairwise relationships between the most impactful features in the dataset. It helps to visualize the distribution and relationships between features.")
pairplot= '../outputs/plots/pairplot.png'
st.image(pairplot, caption='Pairplot of Most Impactful Features', use_container_width=True)

# Handling Class Imbalance
st.write("The class imbalance issue has been addressed using two methoids:")
st.markdown("- **Undersampling**: The majority class is undersampled to balance the dataset.")
st.markdown("- **SMOTE**: Synthetic Data is generated for the minority class to balance the dataset.")
st.write("After evaluating the model performance on both, the undersampled dataset and SMOTE dataset, it was found that the model performed better on the undersampled dataset. Therefore, the undersampled dataset is used for further analysis.")
st.write("This result is as expected since SMOTE is a technique that generates synthetic data which can lead to overfitting, especially when the dataset is already imbalanced. Also, in a critical application like credit card fraud detection, it is important to have a model that has real world data to train on rather than synthetic data.")
# Model Training and Evaluation Details
st.subheader("Model Training and Evaluation")
st.write("The model is trained using a logistic regression algorithm. The dataset is split into training and testing sets, and the model is evaluated using various metrics.")
st.write("1. **Logistic Regression** on Undersampled Dataset")
roc_curve_logistic_undersampled = '../outputs/plots/roc_curve_logistic_regression_undersampled.png'
test_data_logistic_undersampled = '../outputs/plots/test_data_predictions_logistic_regression_undersampled.png'
confusion_matrix_logistic_undersampled = '../outputs/plots/confusion_matrix_logistic_regression_undersampled.png'
st.markdown("#### ROC Curve for Logistic Regression on Undersampled Dataset")
st.image(roc_curve_logistic_undersampled, caption='ROC Curve for Logistic Regression on Undersampled Dataset', width=900)
st.markdown("#### Test Data Predictions for Logistic Regression on Undersampled Dataset")
st.image(test_data_logistic_undersampled, caption='Test Data Predictions for Logistic Regression on Undersampled Dataset', width=900)
st.markdown("#### Confusion Matrix for Logistic Regression on Undersampled Dataset")
st.image(confusion_matrix_logistic_undersampled, caption='Confusion Matrix for Logistic Regression on Undersampled Dataset', width=900)
st.write("The confusion matrix shows the performance of the logistic regression model on the undersampled dataset. The model achieved an accuracy of 0.92, precision of 0.97, recall of 0.86, and F1 score of 0.91.")

st.write("2. **Logistic Regression** on SMOTE Dataset")
roc_curve_logistic_smote = '../outputs/plots/roc_curve_logistic_regression_smote.png'
test_data_logistic_smote = '../outputs/plots/test_data_predictions_logistic_regression_smote.png'
confusion_matrix_logistic_smote = '../outputs/plots/confusion_matrix_logistic_regression_smote.png'
st.markdown("#### ROC Curve for Logistic Regression on SMOTE Dataset")
st.image(roc_curve_logistic_smote, caption='ROC Curve for Logistic Regression on SMOTE Dataset', width=900)
st.markdown("#### Test Data Predictions for Logistic Regression on SMOTE Dataset")
st.image(test_data_logistic_smote, caption='Test Data Predictions for Logistic Regression on SMOTE Dataset', width=900)
st.markdown("#### Confusion Matrix for Logistic Regression on SMOTE Dataset")
st.image(confusion_matrix_logistic_smote, caption='Confusion Matrix for Logistic Regression on SMOTE Dataset', width=900)
st.write("The confusion matrix shows the performance of the logistic regression model on the SMOTE dataset. The model achieved an accuracy of 0.99, precision of 0.13, recall of 0.91, and F1 score of 0.24.")

st.write("3. **Random Forest Classifier** on Undersampled Dataset")
roc_curve_random_forest_undersampled = '../outputs/plots/roc_curve_random_forest_undersampled.png'
test_data_random_forest_undersampled = '../outputs/plots/test_data_predictions_random_forest_undersampled.png'
confusion_matrix_random_forest_undersampled = '../outputs/plots/confusion_matrix_random_forest_undersampled.png'
st.markdown("#### ROC Curve for Random Forest Classifier on Undersampled Dataset")
st.image(roc_curve_random_forest_undersampled, caption='ROC Curve for Random Forest Classifier on Undersampled Dataset', width=900)
st.markdown("#### Test Data Predictions for Random Forest Classifier on Undersampled Dataset")
st.image(test_data_random_forest_undersampled, caption='Test Data Predictions for Random Forest Classifier on Undersampled Dataset', width=900)
st.markdown("#### Confusion Matrix for Random Forest Classifier on Undersampled Dataset")
st.image(confusion_matrix_random_forest_undersampled, caption='Confusion Matrix for Random Forest Classifier on Undersampled Dataset', width=900)
st.write("The confusion matrix shows the performance of the random forest classifier model on the undersampled dataset. The model achieved an accuracy of 0.92, precision of 1.0, recall of 0.83, and F1 score of 0.91.")
st.write("The random forest classifier is not chosen because it has a higher false negative rate than logistic regression. This means that the model is more likely to miss fraudulent transactions, which is not acceptable in a credit card fraud detection system.")

st.write("4. **SVC** on Undersampled Dataset")
roc_curve_svc_undersampled = '../outputs/plots/roc_curve_svc_undersampled.png'
test_data_svc_undersampled = '../outputs/plots/test_data_predictions_svc_undersampled.png'
confusion_matrix_svc_undersampled = '../outputs/plots/confusion_matrix_svc_undersampled.png'
st.markdown("#### ROC Curve for SVC on Undersampled Dataset")
st.image(roc_curve_svc_undersampled, caption='ROC Curve for SVC on Undersampled Dataset', width=900)
st.markdown("#### Test Data Predictions for SVC on Undersampled Dataset")
st.image(test_data_svc_undersampled, caption='Test Data Predictions for SVC on Undersampled Dataset', width=900)
st.markdown("#### Confusion Matrix for SVC on Undersampled Dataset")
st.image(confusion_matrix_svc_undersampled, caption='Confusion Matrix for SVC on Undersampled Dataset', width=900)
st.write("The confusion matrix shows the performance of the SVC model on the undersampled dataset. The model achieved an accuracy of 0.57, precision of 0.51, recall of 0.68, and F1 score of 0.59.")
st.write("The SVC model is not chosen as it has the least performance among all the models. The model is not able to generalize well on the test data and has a high false positive rate. This means that the model is more likely to classify non-fraudulent transactions as fraudulent, which is not acceptable in a credit card fraud detection system.")

st.markdown("### Conclusions")
st.markdown("- The Logistic Regression model performed the best on the undersampled data with an accuracy score of 0.928 and a high true positive rate and less false negative rate. This is the best model for this dataset as it is able to correctly classify the positive and negative classes with a good accuracy score.")
st.markdown("- The SMOTE technique is not preferred for this dataset as it does not provide a good representation of the real data. Since it is a real dataset of a critical problem (Credit Card Fraud Detection), it is better to use the undersampled data for training the model and not to add synthetic data.")
st.markdown("- The Random Forest Classifier performed well on the undersampled data with an accuracy score of 0.928 but had a higher false negative rate than the Logistic Regression model and hence is not chosen.")
st.markdown("- The SVC model has the least accuracy score of 0.57 and is not able to correctly classify the positive and negative classes. This model is not preferred for this dataset.")



