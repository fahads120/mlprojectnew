import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig

@dataclass
class DataIngestionConfig:
    # Configuration for file paths used during data ingestion
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        # Instantiate with default paths from DataIngestionConfig
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        """Reads raw data, saves it, splits it into train/test, and saves those sets."""
        logging.info("Starting data ingestion process")
        try:
            # Read data from source file
            data_path = 'notebook/data/stud.csv'  # Change this path as necessary
            df = pd.read_csv(data_path)
            logging.info(f"Data successfully read from {data_path}")

            # Ensure artifact directory exists
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save the raw data for reference
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info(f"Raw data saved at {self.ingestion_config.raw_data_path}")

            # Split data into training and test sets
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            logging.info("Data split into train and test sets")

            # Save train and test datasets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info(f"Training data saved at {self.ingestion_config.train_data_path}")
            logging.info(f"Testing data saved at {self.ingestion_config.test_data_path}")

            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path

        except Exception as e:
            logging.error("An error occurred during data ingestion", exc_info=True)
            raise CustomException(e, sys)

if __name__ == "__main__":
    # Initialize DataIngestion and run the ingestion process
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()

    # Start data transformation
    data_transformation = DataTransformation()
    data_transformation.initiate_data_transformation(train_data, test_data)
