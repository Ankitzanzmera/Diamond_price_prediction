import sys
import os

sys.path.append(os.getcwd())

from src.logger import logging

if __name__ == "__main__":
    logging.info('In App.py file')