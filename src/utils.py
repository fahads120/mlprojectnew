import os
import sys
import dill
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves an object to a specified file path using dill for serialization.

    Parameters:
    - file_path (str): The path where the object should be saved.
    - obj (Any): The object to save.

    Raises:
    - CustomException: If an error occurs during the saving process.
    """
    try:
        # Ensure the directory for the file path exists
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Save the object using dill
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)

    except Exception as e:
        # Raise a custom exception if there's an error
        raise CustomException(e, sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    """
    Evaluates multiple models using GridSearchCV and calculates R^2 scores.

    Parameters:
    - X_train (pd.DataFrame): Training feature data.
    - y_train (pd.Series or np.array): Training target data.
    - X_test (pd.DataFrame): Testing feature data.
    - y_test (pd.Series or np.array): Testing target data.
    - models (dict): A dictionary where keys are model names and values are model instances.
    - param (dict): A dictionary of hyperparameter grids for each model.

    Returns:
    - dict: A dictionary with model names as keys and their test R^2 scores as values.

    Raises:
    - CustomException: If an error occurs during model evaluation.
    """
    try:
        report = {}

        # Iterate over the models and their corresponding parameters
        for i, model_name in enumerate(models.keys()):
            model = models[model_name]
            param_grid = param[model_name]

            # Perform GridSearchCV to find the best parameters
            gs = GridSearchCV(model, param_grid, cv=3)
            gs.fit(X_train, y_train)

            # Set the best parameters and train the model
            model.set_params(**gs.best_params_)
            model.fit(X_train, y_train)

            # Make predictions on train and test sets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R^2 scores for train and test sets
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)

            # Log the test score
            report[model_name] = test_model_score

        return report

    except Exception as e:
        # Raise a custom exception if there's an error
        raise CustomException(e, sys)
