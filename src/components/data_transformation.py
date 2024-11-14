import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object  # Assuming you have a function to save objects as files
from src.utils import save_object


@dataclass
class DataTransformationConfig:
    # Define the path for the preprocessor object file
    preprocessor_obj_file_path: str = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        """
        Creates and returns a ColumnTransformer object for preprocessing the data.
        This includes scaling numerical features and encoding categorical features.
        """
        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            # Pipeline for numerical features
            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            # Pipeline for categorical features
            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))  # Use with_mean=False for sparse matrices
            ])

            logging.info("Numerical Columns Standard Scaling Completed")
            logging.info("Categorical Columns Encoding Completed")

            # ColumnTransformer to apply pipelines to respective columns
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        """
        Initiates the data transformation process:
        - Reads train and test CSV files
        - Preprocesses the data using the preprocessor object
        - Saves the preprocessor object to a file
        - Returns transformed train and test arrays along with the preprocessor file path
        """
        try:
            # Load train and test data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Loaded train and test data successfully")

            # Obtain the preprocessing object
            preprocessing_obj = self.get_data_transformer_object()

            # Define target and feature columns
            target_column_name = "math_score"
            numerical_columns = ["writing_score", "reading_score"]

            # Separate input features and target for train data
            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            # Separate input features and target for test data
            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on train and test data")

            # Transform training and testing input features
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            # Combine input features and target into single arrays
            train_arr = np.c_[
                input_feature_train_arr, np.array(target_feature_train_df)
            ]
            test_arr = np.c_[
                input_feature_test_arr, np.array(target_feature_test_df)
            ]

            logging.info("Data transformation completed and saving preprocessor object")

            # Save the preprocessor object
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            raise CustomException(e, sys)
