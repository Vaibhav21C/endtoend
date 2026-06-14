import os
import pandas as pd
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import(
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object,evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting the train and test data")
            x_train,y_train,x_test,y_test=(train_array[:,:-1],train_array[:,-1],test_array[:,:-1],test_array[:,-1])
            
            models={
                "LinearRegressor": LinearRegression(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "KNeighborsRegressor": KNeighborsRegressor(),
                "RandomForestRegressor": RandomForestRegressor(),
                "AdaBoostRegressor": AdaBoostRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoostRegressor": CatBoostRegressor(verbose=False)
            }

            model_report:dict=evaluate_model(x_train,y_train,x_test,y_test,models)
            logging.info("model training")

            ##to get the best model score
            best_model_score=max(sorted(model_report.values()))

            ##to get the best model name
            best_model_name=list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model=models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("best model score is less than 0.6")
            logging.info("best model score is greater than 0.6")
            
            save_object(self.model_trainer_config.trained_model_file_path,best_model)    
            predicted=best_model.predict(x_test)
            r2_score_result=r2_score(y_test,predicted)
            return r2_score_result

        except Exception as e:
            logging.info("error in model training")
            raise CustomException(e,sys)    