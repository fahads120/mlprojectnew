import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    """
    Configuration class for defining file paths for data ingestion.
    """
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")


class DataIngestion:
    """
    Handles the ingestion of raw data, including reading, splitting, 
    and saving train/test datasets.
    """
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """
        Executes the data ingestion process:
        - Reads raw data
        - Splits it into training and testing datasets
        - Saves all datasets to specified paths
        """
        logging.info("Starting the data ingestion process.")
        try:
            # Path to raw data
            data_path = 'notebook/data/stud.csv'  # Update this path if necessary
            logging.info(f"Reading data from {data_path}")
            df = pd.read_csv(data_path)

            # Ensure the artifact directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data for reference
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # Split the data into training and testing datasets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data successfully split into train and test sets.")

            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Training data saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Testing data saved at {self.ingestion_config.test_data_path}")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error("An error occurred during data ingestion.", exc_info=True)
            raise CustomException(e, sys)


if __name__ == "__main__":
    logging.info("Starting the pipeline.")

    try:
        # Data ingestion
        ingestion = DataIngestion()
        train_data, test_data = ingestion.initiate_data_ingestion()

        # Data transformation
        logging.info("Starting the data transformation process.")
        data_transformation = DataTransformation()
        train_arr, test_arr, _ = data_transformation.initiate_data_transformation(train_data, test_data)

        # Model training
        logging.info("Starting the model training process.")
        model_trainer = ModelTrainer()
        model_training_results = model_trainer.initiate_model_trainer(train_arr, test_arr)
        print(model_training_results)

    except Exception as e:
        logging.error("Pipeline execution failed.", exc_info=True)
