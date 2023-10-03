import os
import sys
sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    os.makedirs(os.path.join(os.getcwd(),'artifacts'),exist_ok=True)
    preprocessor_path = os.path.join(os.getcwd(),'artifacts','preprocessor.pkl')


class DataTransformation:
    def __init__(self) -> None:
        self.data_transformation_config = DataTransformationConfig()

    def preprocessing_piplines(self,cate_feature,num_feature):
        try:
            self.num_feature = num_feature
            self.cat_feature = cate_feature

            cut_categories = ['Fair','Good','Very Good','Premium','Ideal']
            color_categories = ['D','E','F','G','H','I','J']
            clarity_categories = ['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF']

            cate_pipeline = Pipeline(
                steps=[
                    ('imputer',SimpleImputer(strategy="most_frequent")),
                    ('encoder',OrdinalEncoder(categories=[cut_categories,color_categories,clarity_categories])),
                    ("scaler",StandardScaler())
                ]
            )

            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler())
                ]
            )

            preprocessing_pipeline = ColumnTransformer(
                [
                    ('cate_pipeline',cate_pipeline,cate_feature),
                    ('num_pipeline',num_pipeline,num_feature)
                ],
                remainder="passthrough"
            )

            return preprocessing_pipeline
    
        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,train_data_path,test_data_path):
        try:
            train_data = pd.read_csv(train_data_path)
            test_data = pd.read_csv(test_data_path)
            logging.info('Got the train data and test data')

            target = 'price' 

            train_data_input = train_data.drop([target],axis = 1)
            logging.info(f"{train_data_input.columns}")
            train_data_target = train_data[target]

            test_data_input = test_data.drop([target],axis = 1)
            test_data_target = test_data[target]
            logging.info('Successfully divided data into input and target')

            cat_feature = [feature for feature in train_data_input.columns if train_data_input[feature].dtypes == "object"]
            num_feature = [feature for feature in train_data_input.columns if train_data_input[feature].dtypes != "object"]
            logging.info('Got the Categorical an Numerical Feature')

            preprocessing_pipline = self.preprocessing_piplines(cat_feature,num_feature)
            logging.info('Got the preprocessing Pipeline')

            preprocessed_train_data = preprocessing_pipline.fit_transform(train_data_input)
            preprocessed_test_data = preprocessing_pipline.transform(test_data_input)
            logging.info("Sucessfully transformed data")

            preprocessed_train_data = np.c_[preprocessed_train_data,train_data_target]
            preprocessed_test_data = np.c_[preprocessed_test_data,test_data_target]
            logging.info("Concatenated Input and target data")

            save_object(
                self.data_transformation_config.preprocessor_path,
                preprocessing_pipline
            )

            logging.info('Preprocesser pickle is saved')

            return (preprocessed_train_data,preprocessed_test_data)

        except Exception as e:
            raise CustomException(e,sys)






        




