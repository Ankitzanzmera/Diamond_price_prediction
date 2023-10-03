import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import pymongo as pm
import pickle
from dotenv import load_dotenv

from src.logger import logging
from src.exception import CustomException

host = os.getenv("host")

def read_data_from_mongo():
    
    try:
        client = pm.MongoClient(host)
        db = client['Diamond_Price_Prediction']
        collection = db['data']

        df = pd.DataFrame(list(collection.find()))
        if "_id" in df.columns:
            df = df.drop(["_id"],axis = 1)

        return df
    except Exception as e:
        raise CustomException(e,sys) 

def save_object(filepath,obj):
    try:
        os.makedirs(os.path.dirname(filepath),exist_ok=True)

        with open(filepath,'wb') as file_obj:
            pickle.dump(obj,file_obj)

    except Exception as e:
        raise CustomException(e,sys)
