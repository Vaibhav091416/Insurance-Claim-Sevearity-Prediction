from flask import Flask,request,render_template
import numpy as np
import pandas as pd
from src.logger import logging
from src.exception import CustomException
from src.components.data_transformation import DataTransformation 
from src.utils import load_object
from src.pipeline.predict_pipeline import PredictPipeline

app=Flask(__name__)

@app.route('/')
def index():
    return render_template('index'.html)

@app.route('/predict',methods=['POST','GET'])
def predict_data():
    if request=='GET':
        return render_template('index.html')
    else:
        req_data=request.get_json()
        data=pd.DataFrame(req_data)
        data_id=data['id']

        data=data.drop(columns=['id'],inplace=True)

        pipe=PredictPipeline()
        data=pipe.preprocess(data)
        
        model=load_object('./artifacts/mode.pkl')
        ans=model.predict(data)

        output=pd.DataFrame()
        output['id']=data_id
        output['predicted_loss']=ans 

        output=output.to_json(orient='records')
        return output

if __name__=='__main__':
    app.run(debug=True)
        


