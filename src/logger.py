import os
import logging
from datetime import datetime

dir_name = f'{datetime.now().strftime("%d_%m_%y")}'
DIR_PATH = os.path.join(os.getcwd(),'logs',dir_name)
os.makedirs(DIR_PATH, exist_ok=True)

file_name = f'{datetime.now().strftime("%H_%M_%S")}'
LOG_FILE_PATH = os.path.join(DIR_PATH,file_name)

logging.basicConfig(
    filename = LOG_FILE_PATH,
    level = logging.INFO,
    format = "[ %(asctime)s] - %(lineno)d - %(name)s - %(levelname)s - %(message)s"
)
