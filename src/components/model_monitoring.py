import os,sys
sys.path.append(os.getcwd())

import numpy as np
import mlflow
import mlflow.sklearn
from urllib.parse import urlparse

from src.logger import logging
from src.exception import CustomException
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error


class ModelMonitoring:
    def eval_metrics(self,y_test,y_pred):
        r2 = r2_score(y_test,y_pred)
        mae = mean_absolute_error(y_test,y_pred)
        mse = mean_squared_error(y_test,y_pred)
        rmse = np.sqrt(mse)

        metrics = {"R2_score":r2,"MAE":mae,"MSE":mse,"RMSE":rmse}
        return metrics

    def initiate_model_monitoring(self,tuned_model_name,tuned_model,best_params,X_test,y_test):
        try:
            mlflow.set_registry_uri('https://dagshub.com/Ankitzanzmera/diamond_price_prediction.mlflow')
            tracking_uri_type_score = urlparse(mlflow.get_tracking_uri()).scheme
            logging.info(mlflow.get_tracking_uri())
            logging.info(tracking_uri_type_score)

            with mlflow.start_run():
                y_pred = tuned_model.predict(X_test)

                metrics = self.eval_metrics(y_test,y_pred)

                for i,j in zip(list(metrics.keys()),list(metrics.values())):
                    mlflow.log_metric(i,j)
                logging.info(f"Metrics loaded")

                for i,j in zip(list(best_params.keys()),list(best_params.values())):
                    mlflow.log_param(i,j)
                logging.info('params loaded')

                if tracking_uri_type_score != 'file':
                    mlflow.sklearn.log_model(tuned_model,"Model",registered_model_name=tuned_model_name)
                    logging.info('Model loaded')
                else:
                    logging.info('Error Occurred While Registering Model to MLFLOW')


        except Exception as e:
            raise CustomException(e,sys)
