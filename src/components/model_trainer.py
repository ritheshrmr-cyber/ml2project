import sys
import os
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import custom_exception
from src.logger import logging
from src.utils import save_object
from src.utils import evaluate_models
@dataclass
class modeltrainerconfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class modeltrainer:
    def __init__(self):
        self.model_trainer_config = modeltrainerconfig()
    def initite_model_trainer(self,train_array,test_array):
        try:
            logging.info("splitting training ans test input data")
            x_train,y_train,x_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models={
                "random forest":RandomForestRegressor(),
                "decision tree":DecisionTreeRegressor(),
                "gradient boosting":GradientBoostingRegressor(),
                "linear regressor":LinearRegression(),
                "k-neighbours classifier":KNeighborsRegressor(),
                "xgboost classifier":XGBRegressor(),
                "catboosting classifier":CatBoostRegressor(verbose=False),
                "adaboost classifier":AdaBoostRegressor(),
            }
            params={
                "decision tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "random forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "gradient boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "linear regressor":{},
                "xgboost classifier":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "catboosting classifier":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "adaboost classifier":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "k-neighbours classifier":{
                    "n_neighbors":[5,7,9,11]
                }
                
            }
            model_report :dict=evaluate_models(x_train,y_train,x_test,y_test,models,param = params)
            best_model_score = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]
            if best_model_score<0.6:
                raise custom_exception("no best model found")
            logging.info(f"best found model on both training and testing data")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            predicted=best_model.predict(x_test)
            model_r2_score=r2_score(y_test,predicted)
            return model_r2_score


        except Exception as e:
            raise custom_exception(e,sys)