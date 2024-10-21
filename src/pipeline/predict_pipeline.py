import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        age: int,
        numberOfChildren: int,
        income: int,
        maritalstatus: str,
        educationLevel: str,
        smokingStatus: str,
        physicalActivityLevel: str,
        alcoholConsumption: str,
        dietaryHabits: str,
        sleepPatterns: str,
        historyOfSubstanceAbuse: str,
        familyHistoryDepression: str,
        chronicMedicalConditions: str,
        employmentStatus: str    
        ):

        self.age = age

        self.numberOfChildren = numberOfChildren

        self.income = income

        self.maritalstatus = maritalstatus

        self.educationLevel = educationLevel

        self.smokingStatus = smokingStatus

        self.physicalActivityLevel = physicalActivityLevel

        self.dietaryHabits = dietaryHabits

        self.alcoholConsumption = alcoholConsumption

        self.sleepPatterns = sleepPatterns

        self.historyOfSubstanceAbuse = historyOfSubstanceAbuse

        self.familyHistoryDepression = familyHistoryDepression

        self.chronicMedicalConditions = chronicMedicalConditions

        self.employmentStatus = employmentStatus

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "age": self.age,
                "numberOfChildren": self.numberOfChildren,
                "income": self.income,
                "maritalstatus": self.maritalstatus,
                "educationLevel": self.educationLevel, 
                "smokingStatus": self.smokingStatus, 
                "physicalActivityLevel": self.physicalActivityLevel, 
                "dietaryHabits": self.dietaryHabits,
                "alcoholConsumption": self.alcoholConsumption, 
                "sleepPatterns": [self.sleepPatterns],
                "historyOfSubstanceAbuse": [self.historyOfSubstanceAbuse],
                "familyHistoryDepression": [self.familyHistoryDepression],
                "chronicMedicalConditions": [self.chronicMedicalConditions],
                "employmentStatus": [self.employmentStatus] 
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)