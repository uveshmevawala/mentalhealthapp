from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
import lightgbm as lgb
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
import pickle
import os
import sys
from dataclasses import dataclass
from src.exception import CustomException
from src.logger import logging
from src.components.utils import save_object,evaluate_models

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()


    def initiate_model_trainer(self,x_train_array,y_train_array,x_test_array,y_test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                x_train_array,y_train_array,x_test_array,y_test_array
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                #"Decision Tree": DecisionTreeClassifier(),
                #"Naive Bayes": GaussianNB(),
                #"Logistic Regression": LogisticRegression(),
                "XGBoost": XGBClassifier(),
                "LightGBM": lgb.LGBMClassifier()
            }
            params={
                "Random Forest" : {
                    'n_estimators': [ 50],
                    'max_depth': [ 10],
                    'min_samples_split': [ 5],
                    'min_samples_leaf': [4]
                            },
                "XGBoost" : {
                    'n_estimators': [ 50],
                    'learning_rate': [ 0.2],
                    'max_depth': [5],
                    'subsample': [1.0]
                },
                "LightGBM" : {
                    'n_estimators': [ 50],
                    'learning_rate': [0.2],
                    'num_leaves': [30],
                    'boosting_type': ['dart']
                }
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            print(best_model)

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            f1_s = f1_score(y_test, predicted)
            return f1_s
                        
        except Exception as e:
            raise CustomException(e,sys)