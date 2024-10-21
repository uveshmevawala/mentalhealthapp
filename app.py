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
        return render_template('index.html')
    else:
        data=CustomData(
            age=request.form.get('age'),
            numberOfChildren=request.form.get('numberOfChildren'),
            income=request.form.get('income'),
            maritalstatus=request.form.get('maritalStatus'),
            educationLevel=request.form.get('educationLevel'),
            smokingStatus=request.form.get('smokingStatus'),
            physicalActivityLevel=float(request.form.get('physicalActivityLevel')),
            alcoholConsumption=request.form.get('alcoholConsumption'),
            dietaryHabits=request.form.get('dietaryHabits'),
            sleepPatterns=request.form.get('sleepPatterns'),
            historyOfSubstanceAbuse=request.form.get('historyOfSubstanceAbuse'),
            familyHistoryDepression=float(request.form.get('familyHistoryDepression')),
            chronicMedicalConditions=float(request.form.get('chronicMedicalConditions')),
            employmentStatus=float(request.form.get('employmentStatus'))    
          
        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('index.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")        