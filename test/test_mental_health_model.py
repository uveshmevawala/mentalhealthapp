import unittest
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import pickle

# ### mock class for the mental health prediction model
class MentalHealthModel:
    def __init__(self):
        # placeholder for trained model
        self.model = 'artifacts/model.pkl'
        self.encoder = 'artifacts/transformation_pipeline.pkl'

    def load_model(self, model_file):
        try:
            with open(model_file, 'rb') as f:
                self.model = pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    def preprocess_data(self, df):
        # Assume df is a pandas DataFrame
        if not isinstance(df, pd.DataFrame):
            raise ValueError("Input data must be a pandas DataFrame")

        if 'Age' in df.columns:
            if df['Age'].any() <= 0:
                raise ValueError("Age cannot be negative")

        # Convert Income to financial status category (simple rule-based conversion)
        if 'Income' in df.columns:
            df['Financial Status'] = pd.cut(df['Income'], bins=[0, 30000, 60000, np.inf], 
                                            labels=['Low', 'Middle', 'High'])
        
        # One-hot encode categorical features
        categorical_cols = ['Marital Status', 'Education Level', 'Smoking Status', 'Physical Activity Level']
        self.encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        transformed_data = self.encoder.fit_transform(df[categorical_cols])
        transformed_df = pd.DataFrame(transformed_data, columns=self.encoder.get_feature_names_out(categorical_cols))
        
        return pd.concat([df.drop(columns=categorical_cols), transformed_df], axis=1)

    def predict(self, input_data):
        if self.model is None:
            raise Exception("Model is not loaded")

        # Assume input_data is preprocessed correctly
        prediction = self.model.predict(input_data)
        return prediction

# Unit Test Class
class TestMentalHealthModel(unittest.TestCase):

    def setUp(self):
        # setup the model for testing
        self.model = MentalHealthModel()
        self.model.model = RandomForestClassifier()  # Mock a RandomForestClassifier for testing
        self.model.model.fit(np.random.rand(100, 5), np.random.randint(0, 2, 100))  # Random fitting for testing

    def test_model_load(self):
        
        with self.assertRaises(Exception):
            self.model.load_model('non_existent_file.pkl')

    def test_preprocess_data(self):
        # create a mock dataframe for testing
        df = pd.DataFrame({
            'Age': [25, 30],
            'Marital Status': ['Single', 'Married'],
            'Income': [40000, 70000],
            'Smoking Status': ['Non-smoker', 'Smoker'],
            'Physical Activity Level': ['Active', 'Sedentary'],
            'Education Level': ['PhD']
        })

        processed_df = self.model.preprocess_data(df)

        #####  check if the processed dataframe has the correct number of columns
        expected_columns = [ 'Age', 'Income', 'Financial Status', 
                            'Marital Status', 
                            'Smoking Status'
                            'Physical Activity Level'
                            ]
        self.assertTrue(all(col in processed_df.columns for col in expected_columns))

    def test_age_negative(self):
        # create a Dataframe with a negative age to test edge case
        df = pd.DataFrame({
            
            'Age': [-5],
            'Marital Status': ['Single'],
            'Income': [50000],
            'Smoking Status': ['Non-smoker'],
            'Physical Activity Level': ['Moderate'],
            'Financial Status': ['Welathy'],
            'Education Level': ['PhD']
        })

        # ## test if preprocessing raises the correct exception for negative age
        with self.assertRaises(ValueError) as context:
            self.model.preprocess_data(df)
        self.assertTrue('Age cannot be negative' in str(context.exception))

    def test_invalid_data_type(self):
        # Test if the model raises an exception when passed non-DataFrame input
        with self.assertRaises(ValueError):
            self.model.preprocess_data({'not': 'a dataframe'})

    def test_model_prediction(self):
        # Mock input data
        input_data = pd.DataFrame(np.random.rand(1, 5))  # Random data for prediction
        prediction = self.model.predict(input_data)

        # Check if prediction is in the expected range (0 or 1 for binary classification)
        self.assertIn(prediction[0], [0, 1])

# Edge Case Test Class
class TestEdgeCases(unittest.TestCase):

    def setUp(self):
        self.model = MentalHealthModel()

    def test_missing_columns(self):
        # Test case where expected columns are missing from the input data
        df = pd.DataFrame({
            
            'Income': [50000]  # Missing other fields
        })
        with self.assertRaises(KeyError):
            self.model.preprocess_data(df)

    def test_empty_dataframe(self):
        # Test case with empty DataFrame
        df = pd.DataFrame()
        with self.assertRaises(KeyError):
            self.model.preprocess_data(df)

# Running all the tests
if __name__ == '__main__':
    unittest.main()
