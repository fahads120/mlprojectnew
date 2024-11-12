import logging
import os
from datetime import datetime

# Generate a timestamped log filename
log_filename = datetime.now().strftime('%m_%d_%Y_%H_%M_%S') + ".log"

# Define the path to store logs in a "logs" directory under the current working directory
logs_directory = os.path.join(os.getcwd(), "logs")
os.makedirs(logs_directory, exist_ok=True)  # Create the logs directory if it doesn't exist

# Full path for the log file
log_file_path = os.path.join(logs_directory, log_filename)

# Configure logging with the specified format and level
logging.basicConfig(
    filename=log_file_path,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)

