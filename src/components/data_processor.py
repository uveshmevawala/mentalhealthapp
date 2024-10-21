import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import nltk
from nltk.corpus import names
from nameparser import HumanName
from src.logger import logging

# Download names dataset from nltk
# nltk.download('names')

class DataPreprocessor:
    def __init__(self, df):
        self.df = df
        #self.gender_classifier = self._train_gender_classifier()

    # 1.1 Functionality to generate gender from name
    def _train_gender_classifier(self):
        labeled_names = ([(name, 'male') for name in names.words('male.txt')] +
                         [(name, 'female') for name in names.words('female.txt')])
        featuresets = [(self._name_features(n), gender) for (n, gender) in labeled_names]
        train_set = featuresets
        return nltk.NaiveBayesClassifier.train(train_set)

    def _name_features(self, name):
        return {'last_letter': name[-1].lower()}

    def add_gender_column(self, name_column='Name'):
        self.df['Gender'] = self.df[name_column].apply(self._predict_gender)

    def _predict_gender(self, name):
        try:
            parsed_name = HumanName(name).first
            gender = self.gender_classifier.classify(self._name_features(parsed_name))
            return gender
        except:
            return np.nan

    # 1.2 Convert age to age groups
    def add_age_group_column(self, age_column='Age'):
        bins = [0, 12, 19, 35, 60, np.inf]
        labels = ['Child', 'Teen', 'Adult', 'Middle Aged', 'Senior']
        self.df['AgeGroup'] = pd.cut(self.df[age_column], bins=bins, labels=labels, right=False)

    # 1.3 Convert income to financial status
    def add_financial_status_column(self, income_column='Income'):
        bins = [0, 30000, 100000, np.inf]
        labels = ['Middle Class', 'Upper Class', 'Wealthy']
        self.df['FinancialStatus'] = pd.cut(self.df[income_column], bins=bins, labels=labels, right=False)

    # 2. Convert DataFrame to X, y
    def get_features_and_labels(self, label_column):
        X = self.df.drop(columns=[label_column])
        y = self.df[label_column]
        return X, y

    # 3. One Hot Encoding and Label Encoding
    def encode_categorical_data(self, X):
        # print(X.dtypes)
        categorical_cols = X.select_dtypes(include=['object','category']).columns
        print("cat colsn:\n", categorical_cols)
        encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_encoded = pd.DataFrame(encoder.fit_transform(X[categorical_cols]), columns=encoder.get_feature_names_out())
        X = X.drop(columns=categorical_cols).reset_index(drop=True)
        X_encoded = X_encoded.reset_index(drop=True)
        X_final = pd.concat([X, X_encoded], axis=1)
        return X_final

    def label_encode(self, y):
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        return y_encoded

    # 4. Perform SMOTE
    def apply_smote(self, X, y):
        sm = SMOTE(random_state=42)
        X_res, y_res = sm.fit_resample(X, y)
        return X_res, y_res

    # 5. Train Test Split
    def perform_train_test_split(self, X, y, test_size=0.2):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        return X_train, X_test, y_train, y_test

    # Full pipeline function
    def preprocess(self, label_column):
        # Gender generation from name
        # self.add_gender_column()

        # Age group and financial status generation
        self.add_age_group_column()
        self.add_financial_status_column()

        # Get features and labels
        X, y = self.get_features_and_labels('History of Mental Illness')

        # One hot encode categorical data
        X_encoded = self.encode_categorical_data(X)

        # Label encode target variable
        y_encoded = self.label_encode(y)

        print("X Encoded",X_encoded.iloc[0,:])

        # Apply SMOTE for class imbalance
        X_resampled, y_resampled = self.apply_smote(X_encoded, y_encoded)

        # Train test split
        X_train, X_test, y_train, y_test = self.perform_train_test_split(X_resampled, y_resampled)

        return X_train, X_test, y_train, y_test

# Example usage
# if __name__ == "__main__":
  

#     df = pd.read_csv('artifacts\data.csv')
#     df.drop('Name',axis=1,inplace=True)

#     # Initialize the preprocessor
#     preprocessor = DataPreprocessor(df)

#     # Preprocess the data and split
#     X_train, X_test, y_train, y_test = preprocessor.preprocess('History of Mental Illness')

#     print("X_train:\n", X_train)
#     print("X_test:\n", X_test)
#     print("y_train:\n", y_train)
#     print("y_test:\n", y_test)
