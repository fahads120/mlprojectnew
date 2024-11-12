import sys
import logging
from src.logger import logging

def error_message_detail(error, error_detail: sys) -> str:
    '''
    This function returns a detailed error message with file name, 
    line number, and error description.
    '''
    _, _, exc_tb = error_detail.exc_info()  # Get the traceback object
    file_name = exc_tb.tb_frame.f_code.co_filename  # Get the file name where error occurred
    error_message = "Error occurred in Python script: [{0}] at line number [{1}] with error message: [{2}]".format(
        file_name, exc_tb.tb_lineno, str(error)
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        # Get a detailed error message using the helper function
        self.error_message = error_message_detail(error_message, error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

