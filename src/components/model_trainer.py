import pandas as pd
import numpy as np
import os 
import sys 
from sklearn.ensemble import (RandomForestRegressor,AdaBoostRegressor,GradientBoostingRegressor)
from sklearn.metrics import r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

from src.exception import CustomException
from src.logger import logging

from src.utils import load_object,save_obj,evaluate_models,save_obj

class ModelTrainConfig:
    trained_model_file_path=os.path.join('./artifacts','model.pkl')

class model_train:
    def __init__(self):
        self.model_train_config=ModelTrainConfig()

    def initiate_model_trainer(self,train_arr):
        try:
            t=train_arr.drop(columns=['loss'])
            X_train,X_test,Y_train,Y_test=train_test_split(t,train_arr['loss'],test_size=0.3)
            Y_train=np.log(Y_train)
            Y_test=np.log(Y_test)
            logging.info("Reading in the train test array")

            models={
                "RandomForest":RandomForestRegressor(),
                'AdaBoost':AdaBoostRegressor(),
                "GradientBoost":GradientBoostingRegressor(),
                "DecisionTree":DecisionTreeRegressor(),
            }
            params={
                'RandomForest':{
                    'n_estimators':[40,70,100,200],
                    'max_depth':[10,20]
                },
                "AdaBoost":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "DecisionTree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "GradientBoost":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                }
            }
            model_report:dict=evaluate_models(X_train,X_test,Y_train,Y_test,models,params)
            max_score=max(list(model_report.keys()))
            best_model=list(model_report.keys())[list(model_report.values()).index(max_score)]

            logging.info(f'The best score is {max_score} and the best model is {best_model}')
            save_obj(self.model_train_config.trained_model_file_path,best_model)
            
            predicted=best_model.predict(X_test)
            r2=r2_score(Y_test,predicted)

            print(best_model)
            return r2 

        except Exception as e:
            raise CustomException(e,sys)
            