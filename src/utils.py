import os
import sys
sys.path.append(os.getcwd())
import pandas as pd
import pymongo as pm
import pickle
from dotenv import load_dotenv
from tqdm import tqdm

from src.logger import logging
from src.exception import CustomException

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet    
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR,LinearSVR
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor,AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error 

from sklearn.model_selection import RandomizedSearchCV

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
    
def train_model(X_train,y_train,X_test,y_test):
    try:
        models_list = All_model_list()
        report = {}
        for i in tqdm(range(len(models_list.keys()))):

            model = list(models_list.values())[i]
            model_name = list(models_list.keys())[i]
            model.fit(X_train,y_train)
        
            y_pred = model.predict(X_test)

            score = r2_score(y_test,y_pred)

            report[model_name] = score
            logging.info(f"{model_name} has Done")

        
        logging.info(f'Best Model R2 Without Tuning is {max(report,key=lambda k :report[k])} = {max(report.values())}')
        return report

    except Exception as e:
        raise CustomException(e,sys)


def hyperparameter_tuning(report: dict,X_train,y_train,X_test,y_test):

    model_name = max(report,key= lambda k:report[k])

    logging.info('Doing Hyperparameter Tuning')
    model = All_model_list()[model_name]
    params = get_params_for_model(model_name)

    random_search = RandomizedSearchCV(model, param_distributions=params ,n_iter=2, verbose=False)
    random_search.fit(X_train,y_train)
    best_params_ = random_search.best_params_

    tuned_model = All_model_list()[model_name]
    tuned_model.set_params(**best_params_)
    tuned_model.fit(X_train,y_train)

    y_pred = tuned_model.predict(X_test)

    logging.info(f'After tuning that model Score is {r2_score(y_test,y_pred)}')
    logging.info('Done Hyperparameter Tuning')

    return (model_name,tuned_model,best_params_)

def All_model_list() -> dict:
    model_list = {
        "Linear_Regression":LinearRegression(),
        "Ridge":Ridge(),
        "Lasso":Lasso(),
        "ElasticNet":ElasticNet(),
        "Decision_Tree":DecisionTreeRegressor(),
        "LinearSVM":LinearSVR(),
        "Random_Forest":RandomForestRegressor(),
        "Gradient_Boosting": GradientBoostingRegressor(),
        "AdaBoost":AdaBoostRegressor(),
        "XGB":XGBRegressor(),
        "Catboost":CatBoostRegressor(),
        "Neighbors":KNeighborsRegressor()
    }
    return model_list

def get_params_for_model(model_name: str) -> dict:
    params = {
        "Linear_Regression": {},
        "Ridge": {
            'alpha': [0.01, 0.1, 1.0, 10.0],
        },
        "Lasso": {
            'alpha': [0.01, 0.1, 1.0, 10.0],
        },
        "ElasticNet": {
            'alpha': [0.01, 0.1, 1.0, 10.0],
            'l1_ratio': [0.1, 0.5, 0.9],
        },
        "Decision_Tree": {
            'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        "SVM": {
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'gamma': ['scale', 'auto'] + [0.01, 0.1, 1.0],
        },
        "LinearSVM" :{
            'C': [0.001, 0.01, 0.1, 1, 10, 100],
            'epsilon': [0.01, 0.1, 0.2, 0.5],
            'loss': ['epsilon_insensitive', 'squared_epsilon_insensitive'],
            'dual': [True, False]
        },
        "Random_Forest": {
            'n_estimators': [8, 16, 32, 64, 128, 256],
            'max_depth': [None, 10, 20, 30, 40],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
            'max_features': ['auto', 'sqrt', 'log2'],
        },
        "AdaBoost": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
        },
        "Gradient_Boosting": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 1.0],
            'max_depth': [3, 4, 5],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        },
        "Neighbors": {
            'n_neighbors': [3, 5, 7],
            'weights': ['uniform', 'distance'],
            'p': [1, 2],
        },
        "XGB": {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'max_depth': [3, 4, 5],
            'min_child_weight': [1, 2, 4],
            'subsample': [0.8, 0.9, 1.0],
        },
        "Catboost": {
            'iterations': [50, 100, 200],
            'learning_rate': [0.01, 0.1, 0.3],
            'depth': [4, 6, 8],
            'l2_leaf_reg': [1, 3, 5, 7],
        }
    }
    return params[model_name]