import sys 
import os 
import pickle as pk
import numpy as np
from src.exception import CustomException
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import r2_score
def load_file(file_path):
    try:
        with open(file_path,'r') as file:
            return file.read().strip().split(',')
    except Exception as e:
        raise ConnectionRefusedError(e,sys)

def load_object(file_path):
    try:
        with open(file_path,'rb') as  file:
            return pk.load(file)
    except Exception as e:
        raise CustomException(e,sys)
    
def save_obj(file_path,obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)

        with open(file_path,'wb') as file:
            pk.dump(obj,file)
    except Exception as e:
        raise CustomException(e,sys)
def evaluate_models(X_train,X_test,Y_train,Y_test,models,params):
    try:

        report={}
        print(X_train,Y_train)
        for mod in list(models.keys()):
            model=models[mod]
            para=params[mod]
            print(mod,model,para)
            gs=GridSearchCV(model,para,cv=3)
            gs.fit(X_train,Y_train)
            print("Best paramseters for ",model,**gs.best_params_)

            model.set_params(**gs.best_params_)
            model.fit(X_train,Y_train)

            y_train_pred=model.predict(X_train)
            y_test_pred=model.predict(X_test)

            train_pred_score=r2_score(Y_train,y_train_pred)
            test_pred_score=r2_score(Y_test,y_test_pred)

            report[model]=test_pred_score
        return report
    except Exception as e:
        raise CustomException(e,sys)
            