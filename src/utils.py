import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import pymongo as pm
from dotenv import load_dotenv

from src.logger import logging
from src.exception import CustomException

host = os.getenv("host")

def read_data_from_mongo():
    
    client = pm.MongoClient(host)
    db = client['Diamond_Price_Prediction']
    collection = db['data']

    df = pd.DataFrame(list(collection.find()))
    if "_id" in df.columns:
        df = df.drop(["_id"],axis = 1)

    return df
