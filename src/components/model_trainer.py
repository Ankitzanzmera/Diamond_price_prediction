import os,sys
sys.path.append(os.getcwd())

from src.exception import CustomException
from src.logger import logging
from src.utils import train_model
from src.utils import hyperparameter_tuning
from src.utils import save_object 
from dataclasses import dataclass



@dataclass
class ModelTrainerConfig:
    os.makedirs(os.path.join(os.getcwd(),'artifacts'),exist_ok=True)
    trained_model_path = os.path.join(os.path.join(os.getcwd(),'artifacts','model.pkl'))


class ModelTrainer:
    try:
        def __init__(self) -> None:
            self.model_trainer_config = ModelTrainerConfig()

        def initiate_model_training(self,train_data,test_data):
            X_train,y_train = train_data[:,:-1], train_data[:,-1]
            X_test,y_test = test_data[:,:-1], test_data[:,-1]
            logging.info("Train tests is Splitted")
            logging.info(f'{X_train.shape,y_train.shape,X_test.shape,y_test.shape}')

            report = train_model(X_train,y_train,X_test,y_test)
            tuned_model_name,tuned_model,best_params = hyperparameter_tuning(report,X_train,y_train,X_test,y_test)

            logging.info(f"Best Model Found {tuned_model_name} and params = {best_params}")

            save_object(self.model_trainer_config.trained_model_path,tuned_model)


    except Exception as e:
        raise CustomException(e,sys)
