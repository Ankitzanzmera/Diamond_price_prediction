import os,sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from src.logger import logging
from src.exception import CustomException
from src.utils import load_object


class PredictionPipeline:
    def __init__(self) -> None:
        self.preprocessor_path = os.path.join(os.getcwd(),'artifacts','preprocessor.pkl')
        self.model_path = os.path.join(os.getcwd(),'artifacts','model.pkl')

    def predict(self,input_data):
        try:
            preprocessor_obj = load_object(self.preprocessor_path)
            model_obj = load_object(self.model_path)

            data_scaled = preprocessor_obj.transform(input_data)
            logging.info(f'preprocessor =   {data_scaled}')
            pred = model_obj.predict(data_scaled)
            return pred         
        except Exception as e:
            logging.info("Error occurred at time of predict")
            raise CustomException(e,sys)
        

class CustomData:
    def __init__(self,carat: float,cut: str,color: str,clarity: str,depth: float,table: float,x: float,y: float,z: float):
        self.carat = carat
        self.cut = cut
        self.color = color
        self.clarity = clarity
        self.depth = depth
        self.table = table
        self.x = x
        self.y = y
        self.z = z

    def get_data_as_dataframe(self):
        try:
            logging.info("Making input data as frame")

            input_df = {
                "carat":[self.carat],
                "cut":[self.cut],
                'color':[self.color],
                'clarity':[self.clarity],
                "depth":[self.depth],
                "table":[self.table],
                "x":[self.x],
                "y":[self.y],
                "z":[self.z]
            }
            input_df = pd.DataFrame(input_df)
            logging.info(f'{input_df}')
            logging.info('Input Data Generated for Prediction')
            return input_df
        except Exception as e:
            logging.info('Error occurred at Prediction Pipeline')
            raise CustomException(e,sys)