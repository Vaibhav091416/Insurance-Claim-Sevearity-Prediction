from src.utils import load_object
from src.components.data_transformation import DataTransformation
import pandas as pd

class PredictPipeline:
    def __init__(self):
        self.drop_list=pd.read_csv('./artifacts/columns_to_drop.csv')
        self.transform=DataTransformation()
        self.preprocessor_obj=load_object(self.transform.data_transform_config.preprocessor_path)
    #preprocessing
    def preprocess(self):
        data=self.preprocessor_obj.fit_transform(data)
        data=self.transform.fix_name(data)
        data=self.transform.encode(data)

        #dropping the unwated columns
        drop_list=list(drop_list['Columns'])
        data.drop(columns=drop_list,inplace=True)
        
        return data