from src.components.data_transformation import DataTransformation
from src.components.model_trainer import model_train
from src.logger import logging 
from src.exception import CustomException

if __name__=="__main__":
    train_path='./artifacts/data/train.csv'

    data_transformation=DataTransformation()
    train_arr,_=data_transformation.initiate_transfromation(train_path)
    model_training=model_train()
    print(model_training.initiate_model_trainer(train_arr))