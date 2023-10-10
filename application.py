import sys,os
sys.path.append(os.getcwd())

from flask import Flask,request,render_template,jsonify
from src.logger import logging
from src.pipelines.prediction_pipeline import CustomData,PredictionPipeline
import streamlit as st


application = Flask(__name__)

@application.route("/")
def home_page():
    return render_template('index.html')

@application.route("/predict",methods = ['POST','GET'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('form.html')
    else:
        input_data = CustomData(
            carat = float(request.form.get('carat')),
            cut = str(request.form.get('cut')),
            color = str(request.form.get('color')), 
            clarity = str(request.form.get('clarity')),
            depth = float(request.form.get('depth')),
            table = float(request.form.get('table')),
            x = float(request.form.get('x')),
            y = float(request.form.get('y')),
            z = float(request.form.get('z')),
        )
        final_input_data  = input_data.get_data_as_dataframe()
        logging.info(f'final Input Data = {final_input_data}')
        predict_pipeline = PredictionPipeline()
        pred = predict_pipeline.predict(final_input_data)

        result = round(pred[0],2)
        return render_template('form.html',final_result = result)


if __name__ =="__main__":
    application.run(host = "0.0.0.0",port = 5000,debug=True)