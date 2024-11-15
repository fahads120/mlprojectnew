import os
import sys
from dataclasses import dataclass

# Importing regression models and evaluation metrics
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

# Importing custom modules
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models

@dataclass
class ModelTrainerConfig:
    """
    Configuration for the Model Trainer.
    Stores the path to save the trained model.
    """
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")

class ModelTrainer:
    """
    Handles model training, evaluation, and saving the best model.
    """
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path):
        """
        Trains multiple models, evaluates them, and saves the best-performing model.

        Parameters:
        - train_array (np.array): Training data (features + target).
        - test_array (np.array): Test data (features + target).
        - preprocessor_path (str): Path to the preprocessor object.

        Returns:
        - r2_square (float): R^2 score of the best model on the test dataset.

        Raises:
        - CustomException: For any error during the process.
        """
        try:
            logging.info("Splitting training and test data into features and target variables.")

            # Splitting train and test arrays into features (X) and target (y)
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )

            # Define models to train
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGB Regressor": XGBRegressor(),
                "CatBoost Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            # Evaluate models and get their performance metrics
            logging.info("Evaluating models using training and testing datasets.")
            model_report: dict = evaluate_models(
                X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models
            )

            # Identify the best model based on R^2 score
            best_model_score = max(model_report.values())
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No suitable model found with an acceptable R^2 score.")

            logging.info(f"Best model found: {best_model_name} with R^2 score: {best_model_score}")

            # Save the best model to the specified file path
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            # Predict on test data using the best model
            predicted = best_model.predict(X_test)

            # Calculate R^2 score for test predictions
            r2_square = r2_score(y_test, predicted)
            return r2_square

        except Exception as e:
            # Raise a custom exception for any errors
            logging.error("An error occurred during model training.", exc_info=True)
            raise CustomException(e, sys)
