import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging
import os
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from src.components.utils import save_object
from scipy.sparse import csr_matrix


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
import pickle

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
        
        
        # Add transformed age and income groups to the categorical features
        numerical_to_categorical_transformer = NumericalToCategoricalTransformer()
        print("Num to cat transformation done")
        
        # Pipeline to handle transformation
        self.pipeline = Pipeline(steps=[
            ('numerical_to_categorical', numerical_to_categorical_transformer),  # Step 1: Convert numerical to categorical
            ('one_hot_encoding', OneHotEncoder())  # Step 2: Apply One-Hot Encoding
        ])
        print("setting pipeline fitting")

        # Fit pipeline on input data
        self.pipeline.fit(df)

    def transform_data(self, df):
        """
        Transforms the input dataframe using the previously created pipeline.
        """
        return self.pipeline.transform(df)

    def save_transformation_pipeline(self, filename='transformation_pipeline.pkl'):
        """
        Save the fitted pipeline object as a .pkl file for later use.
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

            # Initialize the DataTransformation object
            transformer = DataTransformation()

            # Create and fit the transformation pipeline
            transformer.create_transformation_pipeline(df)

            # Transform the data
            transformed_data = transformer.transform_data(df)
            print(f"Transformed Data: \n{transformed_data}")

            # Save the transformation pipeline as a pickle file
            preprocessor_obj_file_path = transformer.save_transformation_pipeline()
            print(" transformation save done")

            # Later, you can load the pipeline and use it on test data
            transformer.load_transformation_pipeline()
            print(" transformation load done")

            # Use the loaded pipeline to transform the test data
            # transformed_test_data = transformer.transform_data(df)  # Example: using the same df
            # print(f"Transformed Test Data: \n{transformed_test_data}")

           
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




# @dataclass
# class DataTransformationConfig:
#     preprocessor_obj_file_path = os.path.join('artifacts','preprocessor.pkl')

# class DataTransformation:
#     def __init__(self):
#         self.data_transformation_config = DataTransformationConfig()

#     # def convert_num_to_categorical(self,num_cols):
#     #         print("num_cols",num_cols)
#     #         for cols in num_cols:
#     #             if cols == 'Age':
#     #                 print("cols ",cols)
#     #                 bins = [0, 12, 19, 35, 60, np.inf]
#     #                 labels = ['Child', 'Teen', 'Adult', 'Middle Aged', 'Senior']
#     #                 self.df['AgeGroup'] = pd.cut(self.df[cols], bins=bins, labels=labels, right=False)
#     #             if cols == 'Income':
#     #                 bins = [0, 30000, 100000, np.inf]
#     #                 labels = ['Middle Class', 'Upper Class', 'Wealthy']
#     #                 self.df['FinancialStatus'] = pd.cut(self.df[cols], bins=bins, labels=labels, right=False)
#     #         self.df.drop(num_cols,axis=1,inplace=True)

    
#     def age_to_category(self,Age):
#         return pd.cut(Age, bins=[0, 12, 19, 35, 60, np.inf], labels=['Child', 'Teen', 'Adult', 'Middle Aged', 'Senior'])
    

#     def income_to_category(Self,Income):
#         return pd.cut(Income, bins=[0, 30000, 100000, np.inf], labels=['Middle Class', 'Upper Class', 'Wealthy'])


#     def get_data_transformation_object(self):
#         '''
#         This function is responsible for data transformation.
#         '''
#         try:
#             numerical_to_cat_columns = ["Age", "Income"]
#             #numerical_columns = ["Number of Children"]
#             categorical_columns = [
#                 "Marital Status",
#                 "Education Level",
#                 "Smoking Status",
#                 "Physical Activity Level",
#                 "Employment Status",
#                 "Alcohol Consumption",
#                 "Dietary Habits",
#                 "Sleep Patterns",
#                 "History of Substance Abuse",
#                 "Family History of Depression",
#                 "Chronic Medical Conditions"
#             ]

#             # age_transformer = FunctionTransformer(lambda X: self.age_to_category(X.loc[:,'Age']), validate=False)
#             # income_transformer = FunctionTransformer(lambda X: self.income_to_category(X.loc[:,'Income']), validate=False)

#             preprocessor = ColumnTransformer(
#                 transformers=[
#                     # ('age', age_transformer, ['Age']),
#                     # ('income', income_transformer, ['Income']),
#                     ('cat', OneHotEncoder(), categorical_columns)
#                 ],
#                 remainder='passthrough'  # Leave other columns unchanged
#             )

#             pipeline = Pipeline(steps=[
#                 ('preprocessor', preprocessor)
#                 # ,('onehot', OneHotEncoder(sparse_output=False, handle_unknown='ignore'))
#             ])

            
#             logging.info(f"Categorical columns: {categorical_columns}")
#             logging.info(f"Numerical columns: {numerical_to_cat_columns}")

#             # preprocessor=ColumnTransformer(
#             #     [
#             #     ("num_pipeline",num_pipeline,numerical_to_cat_columns),
#             #     ("cat_pipelines",cat_pipeline,categorical_columns)

#             #     ]
#             # )

#             return preprocessor
        
#         except Exception as e:
#             raise CustomException(e,sys)
        
    
#     def initiate_data_transformation(self,raw_file):
#         try:
#             df = pd.read_csv(raw_file)
#             df.drop('Name',axis=1,inplace=True)

#             label_column = 'History of Mental Illness'

#             X = df.drop(columns=[label_column])
#             y = df[label_column]

#             logging.info(f"X,y split done.")

#             le = LabelEncoder()
#             y_encoded = le.fit_transform(y)

#             logging.info(f"y variable Label encoding  done.")

#             logging.info("Obtaining preprocessing object")

#             preprocessing_obj=self.get_data_transformation_object()

#             X_array=preprocessing_obj.fit_transform(X)

#             save_object(

#                 file_path=self.data_transformation_config.preprocessor_obj_file_path,
#                 obj=preprocessing_obj

#             )

#             sm = SMOTE(random_state=42)
#             X_res, y_res = sm.fit_resample(X_array, y)
            
#             X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#             return (X_train, X_test, y_train, y_test,self.data_transformation_config.preprocessor_obj_file_path)

#         except Exception as e:
#             raise CustomException(e,sys)

