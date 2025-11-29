import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from src.logger import get_logger
from src.custom_exception import CustomException
from config.paths_config import *
from utils.common_functions import read_yaml, load_data

logger=get_logger(__name__)

class DataProcessing:
    def __init__(self,train_path,test_path,processed_path,config_path):
        self.train_path = train_path
        self.test_path = test_path
        self.processed_path = processed_path
        self.config=read_yaml(config_path)

        if not os.path.exists(self.processed_path):
            os.makedirs(self.processed_path)
    def preprocess_data(self,df):
        try:
            logger.info("Starting our Data Processing step")
            logger.info("Dropping the columns")
            df.drop(columns=["Booking_ID","Unnamed: 0"],inplace=True)
            categorical=df.select_dtypes(include=["object"])
            numerical=df.select_dtypes(include=["number"])

            logger.info("Applying Label Encoding")
            le=LabelEncoder()
            for col in categorical:
                df[col]=le.fit_transform(df[col])
            return df

        except Exception as e:
            logger.error(f"Error during preprocess step {e}")
            raise CustomException("Error while preprocess data", e)
    def balance_data(self,df):
        try:
            logger.info("Handling Imbalanced Data")
            X = df.drop(columns='booking_status')
            y = df["booking_status"]

            smote=SMOTE(random_state=42)
            X, y =smote.fit_resample(X,y)

            logger.info("Data balanced sucesffuly")
            return df
        
        except Exception as e:
            logger.error(f"Error during balancing data step {e}")
            raise CustomException("Error while balancing data", e)
    
    def feature_selection(self,df):
        try:
            logger.info("Starting our Feature selection step")

            X = df.drop(columns='booking_status')
            y = df["booking_status"]

            model =  RandomForestClassifier(random_state=42)
            model.fit(X,y)

            feature_importance=model.feature_importances_

            feature_importance_df=pd.DataFrame({"feature" : X.columns,"importance": feature_importance})
            
            top_feature_importance_df=feature_importance_df.sort_values(by="importance",ascending=False)
            print(top_feature_importance_df)

            top_10_features=top_feature_importance_df["feature"].head(10).values

            top_10_df=df[top_10_features.tolist()+["booking_status"]]
            
            logger.info("Feature slection completed sucesfully")

            return top_10_df
        
        except Exception as e:
            logger.error(f"Error during feature selection step {e}")
            raise CustomException("Error while feature selection", e)
    def save_data(self,df , file_path):
        try:
            logger.info("Saving our data in processed folder")

            df.to_csv(file_path, index=False)

            logger.info(f"Data saved sucesfuly to {file_path}")

        except Exception as e:
            logger.error(f"Error during saving data step {e}")
            raise CustomException("Error while saving data", e)
        
    def process(self):
        try:
            logger.info("Loading data from RAW directory")

            train_df = load_data(self.train_path)
            test_df = load_data(self.test_path)

            train_df = self.preprocess_data(train_df)
            test_df = self.preprocess_data(test_df)

            train_df = self.balance_data(train_df)
            test_df = self.balance_data(test_df)

            train_df = self.feature_selection(train_df)
            test_df = test_df[train_df.columns]  

            self.save_data(train_df,PROCESSED_TRAIN_DATA_PATH)
            self.save_data(test_df , PROCESSED_TEST_DATA_PATH)

            logger.info("Data processing completed sucesfully")    
        except Exception as e:
            logger.error(f"Error during preprocessing pipeline {e}")
            raise CustomException("Error while data preprocessing pipeline", e)

if __name__=="__main__":
    processor = DataProcessing(TRAIN_FILE_PATH,TEST_FILE_PATH,PROCESSED_PATH,CONFIG)
    processor.process()  

        
        


        
