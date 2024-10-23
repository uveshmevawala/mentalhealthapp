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

"""
This python file handles all the ML model training.

1) Numerical column gets converted to categorical using NumericalToCategoricalTransformer class.
2) Categorical columns are handled in the DataTransformation class using Pipeline which can be exported as a pickle for data transformer.
2.1) The DataTransformation class does data balancing using SMOTE as the data is imbalanced.
2.2) The DataTransformation class does train test split.

Returns
-------
Transformed data split in X_train,y_train,X_test,y_test.
"""

@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    """
    This function does the heavy lifting work of model training.
    For the demo purpose to make sure that the model training does not take long time, I have considered not so exhaustive list
    of parameters. This will ensure that the model training gets completed quickly. #
    
    The hyperparameter tuning code to trial on computationally expensive parameters is present but commented out. 

    However, during hyperparameter tuning process - I deduced that Random Forest Classifier with 
            300: estimators, 
            20 : max_depth,
            10: min_samples_split,
            10: min_samples_leaf
            gave the best f1_score of 74.89
    """
    def initiate_model_trainer(self,x_train_array,y_train_array,x_test_array,y_test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                x_train_array,y_train_array,x_test_array,y_test_array
            )
            models = {
                "Random Forest": RandomForestClassifier(),
                "XGBoost": XGBClassifier(),
                "LightGBM": lgb.LGBMClassifier()
                #"Decision Tree": DecisionTreeClassifier(),
                #"Naive Bayes": GaussianNB(),
                #"Logistic Regression": LogisticRegression(),
            }
            params={
                "Random Forest" : {
                    'n_estimators': [10],
                    'max_depth': [ 10],
                    'min_samples_split': [ 5],
                    'min_samples_leaf': [4]
                            },
                "XGBoost" : {
                    'n_estimators': [10],
                    'learning_rate': [ 0.2],
                    'max_depth': [5],
                    'subsample': [1.0]
                },
                "LightGBM" : {
                    'n_estimators': [10],
                    'learning_rate': [0.2],
                    'num_leaves': [5],
                    'boosting_type': ['dart']
                }
            }

            ## Code intentionally commented:

            #     models = {
            #     "RandomForest": RandomForestClassifier(),
            #     "XGBoost": XGBClassifier(),
            #     "LightGBM": LGBMClassifier(),
            #     "DecisionTree": DecisionTreeClassifier(),
            #     "GaussianNB": GaussianNB(),
            #     "LogisticRegression": LogisticRegression(solver='liblinear')
            # }
            
            # param_grids = {
            #     "RandomForest": {
            #         'n_estimators': randint(50, 300),
            #         'max_depth': randint(3, 20),
            #         'min_samples_split': randint(2, 10),
            #         'min_samples_leaf': randint(1, 5),
            #         'max_features': ['auto', 'sqrt'],
            #         'bootstrap': [True, False]
            #     },
            #     "XGBoost": {
            #         'n_estimators': randint(50, 300),
            #         'learning_rate': uniform(0.01, 0.2),
            #         'max_depth': randint(3, 10),
            #         'subsample': uniform(0.7, 0.3),
            #         'colsample_bytree': uniform(0.7, 0.3)
            #     },
            #     "LightGBM": {
            #         'n_estimators': randint(50, 300),
            #         'learning_rate': uniform(0.01, 0.2),
            #         'max_depth': randint(3, 10),
            #         'num_leaves': randint(20, 100),
            #         'min_child_samples': randint(10, 50)
            #     },
            #     "DecisionTree": {
            #         'max_depth': randint(3, 20),
            #         'min_samples_split': randint(2, 10),
            #         'min_samples_leaf': randint(1, 5),
            #         'max_features': ['auto', 'sqrt']
            #     },
            #     "GaussianNB": {
            #         'var_smoothing': uniform(1e-9, 1e-8)
            #     },
            #     "LogisticRegression": {
            #         'C': uniform(0.01, 10),
            #         'penalty': ['l1', 'l2'],
            #         'solver': ['liblinear']
            #     }
            # }

            print("Starting model training....")
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
            """
                The reason I chose f1_score is because for the given problem we want to ensure that the model considers the precision 
                as well as recall.

                We don't want to end up in a situation where the person who would get mental illness in future but that person is predicted otherwise.
                So we want to reduce the FN. 
                
                Also we don't want to end up in a situation where model is predicting too high FP cases predicted.
                Hence I chose to go with f1 score.

                We could experiment with Beta value of the f1_score if the business requirement is to focus on reducing FN cases.
            """
            return f1_s
                        
        except Exception as e:
            raise CustomException(e,sys)

            
       

