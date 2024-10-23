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

## Project Structure
```bash
├── artifacts/                      # Directory containing ingested data files, trained ML Model , transformer pkl files .
├── notebooks/                      # Jupyter notebooks for source data, EDA and model experimentation in .ipynb file.
├── src/                            # Python source files (preprocessing, modeling, training, etc.).
|   ├── components                  # Directory containing ML components.
|   |   ├── data_ingestion.py       # Script for ingesting data from source system to local directory.
|   |   ├── data_transformation.py  # Script to transform the source data to make it ready for ML model.
|   |   ├── model_trainer.py        # Script to train the ML models, hyperparameter tuning, evaluate them .
|   ├── pipeline                    # Directory containing scripts to handle prediction.
|   |   ├── predict_pipeline.py     # Script containg all the steps to handle the prediction.
│   ├── exception.py                # Script to handle exceptions & errors.
│   ├── logger.py                   # Script for logging.
├── test/                           # Directory containing the unit test cases .
|   ├── test_mental_health_model.py # .py containing the unit test cases .
├── templates/                      # Directory containing html files for app UI .
|   ├── index.html                  # html file of the index file.
|   ├── home.html                   # html file of the app homepage.
├── README.md                       # Project overview and instructions.
├── requirements.txt                # Dependencies and libraries needed.
├── app.py                          # Flask app file - Main script to run the application.
└── setup.py                        # file to make the app as library/package to be deployed in PyPI.
```

## Project Structure

### 1. Clone the repository:

```bash
git clone https://github.com/uveshmevawala/mentalhealthapp.git
```

### 2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### 3. Running the Application

```bash
python app.py
```







