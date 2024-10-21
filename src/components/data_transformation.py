import sys
from dataclasses import dataclass

import numpy as np 
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import gender_guesser.detector as gender

from src.exception import CustomException
from src.logger import logging
import os

# from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"proprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function si responsible for data trnasformation
        
        '''
        try:


            numerical_columns = ["Age", "Income"]
            categorical_columns = [
                "Marital Status",
                "Education Level",
                "Number of Children",
                "Smoking Status",
                "Physical Activity Level",
                "Employment Status",
                "Alcohol Consumption",
                "Dietary Habits",
                "Sleep Patterns",
                "History of Substance Abuse",
                "Family History of Depression",
                "Chronic Medical Conditions",
                "Gender",
                "income_groups",
                "age_groups"
            ]

            num_pipeline= Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())

                ]
            )

            cat_pipeline=Pipeline(

                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
                ]

            )

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            preprocessor=ColumnTransformer(
                [
                ("num_pipeline",num_pipeline,numerical_columns),
                ("cat_pipelines",cat_pipeline,categorical_columns)

                ]


            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transformation(self,raw_path):
        try:
            # raw_df=pd.read_csv(raw_path)
                        
            logging.info("Read raw data completed")

            preprocessing_obj=self.initiate_eda(raw_path)

            X = raw_df.drop(['History of Mental Illness',], axis = 1)
            y = raw_df['History of Mental Illness']

            logging.info("Obtaining preprocessing object")

            

            preprocessing_obj=self.get_data_transformer_object()

        except Exception as e:
            raise CustomException(e,sys)
        
    def get_first_name(self, fullname):
        firstname = ''
        try:
            firstname = fullname.split()[0] 
        except Exception as e:
            print(str(e))
        return firstname
        
    def generate_gender(self,data):
        try:
            gd = gender.Detector()
            pat=r'(\,|\.|Mrs.|Jr.|Dr.|Mr.|Miss|Ms)'
            data['Name'].replace(pat,'',regex=True, inplace=True)
            data['Fname'] = data['Name'].map(lambda x: get_first_name(x))
            data['Gender'] = data['Fname'].map(lambda x: gd.get_gender(x))
            data['Gender'].replace('mostly_female', 'female', inplace=True)
            data['Gender'].replace('mostly_male', 'male', inplace=True)
            data['Gender'].replace('andy', 'female', inplace=True)
            data['Gender'].replace('unknown', 'female', inplace=True)
        except Exception as e:
            print(str(e))
        return data
            

    def age_groups(self,age):
        try:
            if age<13:
                return "Children"
            elif  (age >13) & (age<=17):
                return "Teenagers"
            elif (age > 17) & (age<24):
                return "Young Adults"
            elif (age > 24) & (age< 34):
                return "Adults"
            elif (age > 34) & (age< 60):
                return "Elderly"
            else:
                return "Retired"
        except Exception as e:
            print(str(e))
        
    def financial_status(income):
        try:
            if income < 15000:
                return "Poor"
            elif income< 45000 :
                return "Lower Middle Class"
            elif income < 120000:
                return "Middle Class"
            elif income < 160100:
                return "Upper Class"
            else:
                return "Wealthy"
        except Exception as e:
            print(str(e))
    
    def initiate_eda(self,raw_path):
        try:
            raw_df=pd.read_csv(raw_path)

            df = generate_gender(raw_df)

            df["age_groups"] = df["Age"].apply(lambda x:age_groups(x))

            df["income_groups"] = df["Income"].apply(lambda x:financial_status(x))

        except Exception as e:
            raise CustomException(e,sys)
        
        return df