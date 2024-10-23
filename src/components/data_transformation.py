import sys
from dataclasses import dataclass
import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from src.components.utils import save_object
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

"""
This python file handles all the data transformation required for the ML model.

1) Numerical column gets converted to categorical using NumericalToCategoricalTransformer class.
2) Categorical columns are handled in the DataTransformation class using Pipeline which can be exported as a pickle for data transformer.
2.1) The DataTransformation class does data balancing using SMOTE as the data is imbalanced.
2.2) The DataTransformation class does train test split.

Returns
-------
Transformed data split in X_train,y_train,X_test,y_test.
"""

# Custom transformer to convert numerical columns (like age, income) into categorical values
class NumericalToCategoricalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, age_bins=None, income_bins=None):
        self.age_bins = age_bins or [0, 12, 18, 35, 60, np.inf]  # Default age bins
        self.age_labels = ['Child', 'Teen', 'Young Adult', 'Adult', 'Senior']
        self.income_bins = income_bins or [0, 30000, 60000, 100000, np.inf]  # Default income bins
        self.income_labels = ['Low', 'Middle', 'Upper Middle', 'Wealthy']
        print("In NumericalToCategoricalTransformer func")

    def fit(self, X, y=None):
        return self  # No fitting necessary

    def transform(self, X, y=None):
        X = X.copy()
        print("In transform func")
        # Convert age and income columns to categorical
        X['age_group'] = pd.cut(X['Age'], bins=self.age_bins, labels=self.age_labels)
        X['income_group'] = pd.cut(X['Income'], bins=self.income_bins, labels=self.income_labels)
        return X[['age_group', 'income_group'] + [col for col in X.columns if col not in ['Age', 'Income']]]

# DataTransformation class to perform both categorical and numerical transformations
class DataTransformation:
    def __init__(self):
        self.ohe = None  # Placeholder for OneHotEncoder object
        self.pipeline = None  # Placeholder for transformation pipeline

    def create_transformation_pipeline(self, df):
        """
        Creates a pipeline that transforms numerical columns (age, income)
        and performs one-hot encoding on categorical features.
        """
        # Define which columns are categorical and numerical
        print("In create_transformation func")
        categorical_features = [
                    "Marital Status",
                    "Education Level",
                    "Smoking Status",
                    "Physical Activity Level",
                    "Employment Status",
                    "Alcohol Consumption",
                    "Dietary Habits",
                    "Sleep Patterns",
                    "History of Substance Abuse",
                    "Family History of Depression",
                    "Chronic Medical Conditions"
                ]
        
        
        ##### add transformed age and income groups to the categorical features
        numerical_to_categorical_transformer = NumericalToCategoricalTransformer()
        print("Num to cat transformation done")
        
        ### pipeline to handle transformation
        self.pipeline = Pipeline(steps=[
            ('numerical_to_categorical', numerical_to_categorical_transformer),  # Step 1: Convert numerical to categorical
            ('one_hot_encoding', OneHotEncoder())  # Step 2: Apply One-Hot Encoding
        ])
        print("setting pipeline fitting")

        self.pipeline.fit(df)

    def transform_data(self, df):
        """
        transforms the input dataframe using the previously created pipeline.
        """
        return self.pipeline.transform(df)

    def save_transformation_pipeline(self, filename='artifacts/transformation_pipeline.pkl'):
        """
        save the fited pipeline object as a .pkl file for later use.
        """
        with open(filename, 'wb') as file:
            pickle.dump(self.pipeline, file)
        print(f"Transformation pipeline saved as {filename}")
        return filename

    def load_transformation_pipeline(self, filename='transformation_pipeline.pkl'):
        """
        Load the saved pipeline object for transforming test data.
        """
        with open(filename, 'rb') as file:
            self.pipeline = pickle.load(file)
        print(f"Transformation pipeline loaded from {filename}")

    def initiate_data_transformation(self,raw_file):
        try:
            df = pd.read_csv(raw_file)

            label_column = 'History of Mental Illness'
            y = df[label_column]

            df.drop(['Name','History of Mental Illness'],axis=1,inplace=True)

            X = df.copy()
            

            logging.info(f"X,y split done.")

            le = LabelEncoder()

            y_encoded = le.fit_transform(y)

            logging.info(f"y variable Label encoding  done.")

            logging.info("Obtaining preprocessing object")

            ### initialize the DataTransformation object
            transformer = DataTransformation()

            # create and fit the transformation pipeline
            transformer.create_transformation_pipeline(df)

            ## Transform the data
            transformed_data = transformer.transform_data(df)
            print(f"Transformed Data: \n{transformed_data}")

            # save the transformation pipeline as a pickle file
            preprocessor_obj_file_path = transformer.save_transformation_pipeline()
            print(" transformation save done")

            # Later, you can load the pipeline and use it on test data
            transformer.load_transformation_pipeline()
            print(" transformation load done")

            transformed_data_array = transformed_data.toarray()
            print("transform data shape",transformed_data.shape)
            print("y_encoded data shape",y_encoded.shape)
            print("type y_encoded ",type(y_encoded))
            print("type transformed_data ",type(transformed_data))
            print("type transformed_data ",type(transformed_data_array))

            sm = SMOTE(sampling_strategy='minority',random_state=42)
            X_res, y_res = sm.fit_resample(transformed_data_array, y_encoded)
            print(" doing smote ")
            
            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
            return (X_train, X_test, y_train, y_test,preprocessor_obj_file_path)

        except Exception as e:
            raise CustomException(e,sys)


