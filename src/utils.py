import os
import sys
import dill
import pandas as pd
import numpy as np
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
