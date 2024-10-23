## Mental Health Prediction App

## Overview

This project is a Machine Learning application designed to predict mental illness based on lifestyle factors. The goal is to develop an accurate and efficient model for prediction.

## Features
### Data Preprocessing: 
Categorical and numerical feature transformations, one-hot encoding, label encoding, and handling missing values.
### Modeling: 
Uses multiple classification algorithms (Random Forest, XGBoost, Decision Trees) with hyperparameter tuning via GridSearch/RandomizedSearch.
### Evaluation: 
Evaluates model performance using accuracy, F1-score, precision, recall, and confusion matrix.
### SMOTE: 
Implements SMOTE for handling class imbalance.
### Model Deployment:
 Best performing model saved as a .pkl file.

## Prerequisites
To run this project, ensure you have the following dependencies installed:

Python 3.8+
Pandas
NumPy
Scikit-learn
XGBoost
LightGBM
Matplotlib / Seaborn (for data visualization)
SMOTE (imbalanced-learn library)
etc. The full list of libraries are present in requirements.txt file.

## Install dependencies using the command:

pip install -r requirements.txt

