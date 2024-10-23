import os
import sys
from src.exception import CustomException
from src.logger import logging

import pandas as pd

from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer


@dataclass
class DataIngestionConfig:
    raw_data_path: str=os.path.join('artifacts','data.csv')

class DataIngestion:
    """
    A class used to Ingest the data.
    
    Methods
    -------
    initiate_data_ingestion():
        reads the ingested file.
    """
        
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Ingests the data from the source & stores in local directory .
        
        Parameters:
        
        
        Returns:
        Local directory location where file is saved.
        """
        logging.info("Entered the data ingestion method or component")
        try:
            df=pd.read_csv('notebook\data\depression_data.csv')
            logging.info('Read the dataset as dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)


            logging.info("Ingestion of the data iss completed")

            return(
                self.ingestion_config.raw_data_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    #  Initialize the data ingestion
    obj=DataIngestion()
    raw_data=obj.initiate_data_ingestion()

    #  Initialize the preprocessor
    preprocessor = DataTransformation()  
    X_train, X_test, y_train, y_test,_=preprocessor.initiate_data_transformation(raw_data) 
    print("Data transformation done....")

    #  Initialize & run the model trainer          
    modeltrainer=ModelTrainer()
    print(modeltrainer.initiate_model_trainer(X_train,y_train,X_test,y_test))




