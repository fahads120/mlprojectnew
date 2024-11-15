import os
import sys
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    """
    Saves an object to a specified file path using pickle.

    Parameters:
    - file_path (str): Path where the object should be saved.
    - obj (Any): The object to be saved.

    Raises:
    - CustomException: If an error occurs during the saving process.
    """
    try:
        # Create directory if it doesn't exist
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        # Log the saving process
        logging.info(f"Saving object to {file_path}")

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

        logging.info(f"Object saved successfully to {file_path}")

    except Exception as e:
        raise CustomException(f"Error saving object to {file_path}: {str(e)}", sys)

def load_object(file_path):
    """
    Loads an object from a specified file path using pickle.

    Parameters:
    - file_path (str): Path of the file to load the object from.

    Returns:
    - Any: The object loaded from the file.

    Raises:
    - CustomException: If an error occurs during the loading process.
    """
    try:
        logging.info(f"Loading object from {file_path}")

        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        logging.info(f"Object loaded successfully from {file_path}")
        return obj

    except Exception as e:
        raise CustomException(f"Error loading object from {file_path}: {str(e)}", sys)

def evaluate_models(X_train, y_train, X_test, y_test, models, params):
    """
    Evaluates multiple models using GridSearchCV and calculates their R^2 scores.

    Parameters:
    - X_train (np.array): Training feature set.
    - y_train (np.array): Training target values.
    - X_test (np.array): Test feature set.
    - y_test (np.array): Test target values.
    - models (dict): Dictionary of model names as keys and model instances as values.
    - params (dict): Dictionary of model names as keys and hyperparameters as values.

    Returns:
    - dict: A dictionary containing model names as keys and their R^2 scores as values.

    Raises:
    - CustomException: If an error occurs during the evaluation process.
    """
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Starting evaluation for model: {model_name}")

            # Fetch hyperparameters for the current model
            model_params = params.get(model_name, {})

            # Perform hyperparameter tuning using GridSearchCV
            logging.info(f"Starting GridSearchCV for model {model_name} with params {model_params}")
            grid_search = GridSearchCV(estimator=model, param_grid=model_params, cv=3)
            grid_search.fit(X_train, y_train)

            # Set the best parameters to the model
            model.set_params(**grid_search.best_params_)
            model.fit(X_train, y_train)

            # Predict on training and test datasets
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate R^2 scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            logging.info(f"{model_name}: Train R^2: {train_score}, Test R^2: {test_score}")

            # Add the model's test score to the report
            report[model_name] = test_score

        logging.info("Model evaluation completed successfully.")
        return report

    except Exception as e:
        raise CustomException(f"Error during model evaluation: {str(e)}", sys)
