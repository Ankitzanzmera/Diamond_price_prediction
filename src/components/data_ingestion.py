import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.model_selection import train_test_split

from src.logger import logging
from src.exception import CustomException
from src.utils import read_data_from_mongo

@dataclass
class DataIngestionConfig:
    os.makedirs(os.path.join(os.getcwd(),'artifacts'),exist_ok=True)
    raw_data_path = os.path.join(os.getcwd(),'artifacts','raw.csv')
    train_data_path = os.path.join(os.getcwd(),'artifacts','train.csv')
    test_data_path = os.path.join(os.getcwd(),'artifacts','test.csv')

class DataIngestion:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()

    def initiate_data_injestion(self):
        try:
            logging.info('Reading Data From Mongodb')
            df = read_data_from_mongo()

            df.to_csv(self.data_ingestion_config.raw_data_path,header = True,index=False)

            (train_data,test_data) = train_test_split(df,test_size=0.3,random_state=45,shuffle=True)
            logging.info('Train Test Splitted')

            train_data.to_csv(self.data_ingestion_config.train_data_path,header = True,index = False)
            test_data.to_csv(self.data_ingestion_config.test_data_path,header = True,index = False)
            logging.info('Saved Train And Test to Artifacts')

            return(
                self.data_ingestion_config.train_data_path,
                self.data_ingestion_config.test_data_path
            )

        except Exception as e:
            raise CustomException(e,sys)
