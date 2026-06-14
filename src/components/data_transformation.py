import sys 
from dataclasses import dataclass

import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.utils import save_object
from src.exception import CustomException
from src.logger import logging
import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path:str=os.path.join('artifacts',"preprocessor.pkl")
  
class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()  
    
    def get_data_transformation_object(self):
        try: 
            logging.info('Data Transformation started') 
            numerical_columns=['writing_score','reading_score']
            categorical_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']
            
            num_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='median')),
                                        ('scaler',StandardScaler())])
            
            cat_pipeline=Pipeline(steps=[('imputer',SimpleImputer(strategy='most_frequent')),
                                        ('onehotencoder',OneHotEncoder(handle_unknown='ignore')),
                                        ('scaler',StandardScaler(with_mean=False))])
            
            logging.info("preprocessing pipeline completed")

            preprocessor=ColumnTransformer(transformers=[('num_pipeline',num_pipeline,numerical_columns),
                                                        ('cat_pipeline',cat_pipeline,categorical_columns)])
            
            logging.info("Column Transformer created")
            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            logging.info("Read the datasets")
            logging.info("Obtaining preprocessing object")

            preprocessor_obj=self.get_data_transformation_object()    
            target_column_name='math_score'

            numerical_columns=['writing_score','reading_score']
            input_features_train_df=train_df.drop(columns=[target_column_name],axis=1)
            input_features_test_df=test_df.drop(columns=[target_column_name],axis=1)
            
            train_target_feature=train_df[target_column_name]
            test_target_feature=test_df[target_column_name]

            logging.info(f"Applying preprocessing")

            input_features_train_array=preprocessor_obj.fit_transform(input_features_train_df)
            input_features_test_array=preprocessor_obj.transform(input_features_test_df)

            logging.info("preprocessing")
            train_arr=np.c_[input_features_train_array,np.array(train_target_feature)]
            test_arr=np.c_[input_features_test_array,np.array(test_target_feature)]

            logging.info(f"saving the preprocessor object")

            save_object(obj=preprocessor_obj,file_path=self.data_transformation_config.preprocessor_obj_file_path)
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            ) 

        except Exception as e:
            raise CustomException(e,sys)

  