from src.logger import logging
from src.exception import CustomException
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from src.utils import load_file
from src.utils import load_object,save_obj
import pandas as pd
import numpy as np
import sys
import os

class data_transform_config:
    def __init__(self):
        self.preprocessor_path=os.path.join('./artifacts','preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transform_config=data_transform_config()
        self.cont_data=load_file('./artifacts/cont_data.txt')
        self.cat_data=load_file('./artifacts/cat_data.txt')
    def get_transformer_obj(self):
        try:
            numerical_pipeline=Pipeline(steps=[
                ("Imputer",SimpleImputer(strategy='median'))
            ])
            categorical_pipeline=Pipeline(
                steps=[
                    ('Imputer',SimpleImputer(strategy='most_frequent'))
                ]
            )
            logging.info(f"Numerical column:{self.cont_data}")
            logging.info(f"Categorical column:{self.cat_data}")

            preprocessor=ColumnTransformer([
                ('num_pipeline',numerical_pipeline,self.cont_data),
                ('categorical_data',categorical_pipeline,self.cat_data)
            ])

            preprocessor.set_output(transform='pandas')

            logging.info("Returning Preprocessor Object.")
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
    
    def encode(self,data):
        for lis in self.cat_data:
            obj=load_object('./artifacts/label_encoders/'+str(lis)+'.pkl')
            data[lis]=obj.transform(data[lis])
            data[lis]=data[lis].astype('category',copy=False)

        return data
    def fix_name(self,data):
        col=list(data.columns)
        dk={}
        for c in col:
            dk[c]=c.split('__')[1]

        data.rename(columns=dk,inplace=True)
        return data

        
    def initiate_transfromation(self,train_path):
        try:
            train=pd.read_csv(train_path)
            # test=pd.read_csv(test_path)

            logging.info("Reading training and testing data.")

            preprocessing_obj=self.get_transformer_obj()
            cont_data=load_file('./artifacts/cont_data.txt')
            cat_data=load_file('./artifacts/cat_data.txt')
            target='loss'
            exclude=[target,'id']

            input_feature_train_df=train.drop(columns=exclude,axis=1)
            target_feature_train_df=train[target]

            # input_feature_test_df=test.drop(columns=exclude,axis=1)
            # target_feature_test_df=test[target]

            logging.info("Applying Preprocessing to train and test data")
            print(type(input_feature_train_df)) 
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)


            # input_feature_test_arr=preprocessing_obj.fit_transform(input_feature_test_df)
            self.fix_name(input_feature_train_arr)
            self.encode(input_feature_train_arr)

            drop_df=pd.read_csv('./artifacts/columns_to_drop.csv')
            drop_list=list(drop_df['Columns'])
            
            final_feature_train=input_feature_train_arr.drop(columns=drop_list)
            # final_feature_test=input_feature_test_arr.drop(columns=drop_list)


            train_x=pd.concat([final_feature_train,target_feature_train_df],axis=1)
            # test_x=pd.concat([final_feature_test,target_feature_test_df],axis=1)

            save_obj(self.data_transform_config.preprocessor_path,preprocessing_obj)


            return (train_x,self.data_transform_config.preprocessor_path)
        except Exception as e:
            raise CustomException(e,sys)





