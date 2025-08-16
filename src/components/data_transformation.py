import sys
from dataclasses import dataclass

import numpy as np  # type: ignore
import pandas as pd # type: ignore
from sklearn.compose import ColumnTransformer # type: ignore
from sklearn.impute import SimpleImputer # type: ignore
from sklearn.pipeline import Pipeline # type: ignore
from sklearn.preprocessing import OneHotEncoder, StandardScaler # type: ignore

from src.exception import CustomException
from src.logger import logging
import os

from src.utils import save_object

# Dataclass to store configuration for saving the preprocessor object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")  
    # Path where the preprocessor object will be saved

class DataTransformation:
    def __init__(self):
        # Create an instance of DataTransformationConfig so we can access the file path
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        Creates and returns a preprocessing object for transforming the dataset.
        This object will handle missing values, scaling, and encoding.
        '''
        try:
            # Columns that contain numbers
            numerical_columns = ["writing_score", "reading_score"]

            # Columns that contain text categories
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]

            # Pipeline for numerical columns
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),  # Fill missing numbers with median
                    ("scaler", StandardScaler())  # Scale numbers so mean=0, variance=1
                ]
            )

            # Pipeline for categorical columns
            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing text with most common value
                    ("one_hot_encoder", OneHotEncoder()),  # Convert categories into binary columns
                    ("scaler", StandardScaler(with_mean=False))  # Scale values but don't center them (needed for sparse data)
                ]
            )

            # Log which columns are being processed
            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            # Combine both pipelines into one preprocessor
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),  # Apply num_pipeline to numerical columns
                    ("cat_pipelines", cat_pipeline, categorical_columns)  # Apply cat_pipeline to categorical columns
                ]
            )

            return preprocessor  # Return the combined preprocessor
        
        except Exception as e:
            # If something goes wrong, raise a custom exception
            raise CustomException(e, sys)
        
    def initiate_data_transformation(self, train_path, test_path):
        '''
        Reads train and test CSV files, applies preprocessing,
        and saves the preprocessor for later use.
        '''
        try:
            # Read train and test CSV files into pandas DataFrames
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("Obtaining preprocessing object")

            # Get the preprocessor object we defined earlier
            preprocessing_obj = self.get_data_transformer_object()

            # The column we are trying to predict
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features (X) and target (y) for training data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features (X) and target (y) for testing data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(
                "Applying preprocessing object on training dataframe and testing dataframe."
            )

            # Fit the preprocessor on training inputs and transform them fit() → Learn the parameters from the data (e.g., mean & std for scaling, categories for encoding).
            

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            # Transform test inputs (no fit here to prevent data leakage)transform() → Apply those learned parameters to convert the data.
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine processed features with the target column for train and test np.c_ converts to coloumn-wise arrays
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object.")

            # Save the preprocessor object to a file for future use
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            # Return the processed train array, test array, and preprocessor file path
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)
