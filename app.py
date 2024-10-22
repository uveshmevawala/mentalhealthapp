from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('index.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            age=int(request.form.get('age')),
            maritalstatus=request.form.get('maritalStatus'),
            educationLevel=request.form.get('educationLevel'),
            numberOfChildren=int(request.form.get('numberOfChildren')),
            smokingStatus=request.form.get('smokingStatus'),
            physicalActivityLevel=request.form.get('physicalActivityLevel'),
            employmentStatus=request.form.get('employmentStatus'),
            income=int(request.form.get('income')),
            alcoholConsumption=request.form.get('alcoholConsumption'),
            dietaryHabits=request.form.get('dietaryHabits'),
            sleepPatterns=request.form.get('sleepPatterns'),
            historyOfSubstanceAbuse=request.form.get('historyOfSubstanceAbuse'),
            familyHistoryDepression=request.form.get('familyHistoryDepression'),
            chronicMedicalConditions=request.form.get('chronicMedicalConditions')
                
          
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        if results[0] == 0:
            message = "This person will not suffer from mental illness."
        elif results[0] == 1:
            message = "This person will suffer from mental illness."

        print("after Prediction")
        return render_template('home.html',results=message)
    

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)        