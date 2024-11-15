import os
import sys
from dataclasses import dataclass
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor, GradientBoostingRegressor, RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray) -> float:
        """
        Trains multiple models, evaluates their performance, and saves the best-performing model.
        
        Parameters:
        - train_array (np.ndarray): Training data (features + target).
        - test_array (np.ndarray): Test data (features + target).
        
        Returns:
        - r2_square (float): R^2 score of the best model on the test dataset.
        
        Raises:
        - CustomException: If any error occurs during the model training or evaluation.
        """
        try:
            logging.info("Splitting training and testing datasets into features and targets.")
            
            # Split the train and test arrays into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],  # Features from the training set
                train_array[:, -1],   # Target from the training set
                test_array[:, :-1],   # Features from the test set
                test_array[:, -1],    # Target from the test set
            )

            # Define regression models to train
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGB Regressor": XGBRegressor(),
                "KNN Regressor": KNeighborsRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor(),
                "CatBoost Regressor": CatBoostRegressor(silent=True),
            }

            # Define hyperparameters for each model
            params = {
                "Random Forest": {"n_estimators": [100, 200], "max_depth": [10, 20]},
                "Decision Tree": {"max_depth": [10, 20, None]},
                "Gradient Boosting": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
                "XGB Regressor": {"n_estimators": [100, 200], "learning_rate": [0.01, 0.1]},
                "KNN Regressor": {"n_neighbors": [5, 10]},
                "AdaBoost Regressor": {"n_estimators": [50, 100]},
                "CatBoost Regressor": {"iterations": [100, 200], "learning_rate": [0.01, 0.1]},
            }

            # Pass both models and params to evaluate_models
            model_report = evaluate_models(
                X_train, y_train, X_test, y_test, models, params
            )
            best_model_name = min(model_report, key=model_report.get)
            best_model = models[best_model_name]

            # Save the best model
            save_object(self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(f"Best model: {best_model_name} has been saved at {self.model_trainer_config.trained_model_file_path}")
            
            # Return R^2 score of the best model
            r2_square = model_report[best_model_name]
            return r2_square

        except Exception as e:
            raise CustomException(e, sys)
