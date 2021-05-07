import logging
import sys

from pathlib import Path
from typing import Union


def clean_log(log_name: str = "root"):
    open(f"./logs/{log_name}.log", "w").close()


# https://stackoverflow.com/questions/54591352/python-logging-new-log-file-each-loop-iteration
def get_custom_logger(logger_name: Union[str, None], root_logger_path: Path = Path("./logs/"), level=logging.DEBUG):
    """
    Method to return a custom logger with the given name and level
    """
    if not root_logger_path.exists():
        root_logger_path.mkdir()
    logger = logging.getLogger(logger_name)
    if not logger_name:
        logger_name = "root.log"
    logger.handlers = []
    logger.setLevel(level)
    format_string = ("%(asctime)s — %(name)s — %(levelname)s — %(funcName)s:"
                     "%(lineno)d — %(message)s")
    log_format = logging.Formatter(format_string)
    # Creating and adding the file handler
    file_handler = logging.FileHandler(root_logger_path.joinpath(f"{logger_name}"), mode='w')
    file_handler.setFormatter(log_format)
    logger.addHandler(file_handler)
    # Creating and adding a full file handler
    file_handler_full = logging.FileHandler(root_logger_path.joinpath(f"root_complete.log"), mode='w')
    file_handler_full.setFormatter(log_format)
    logger.addHandler(file_handler_full)
    # Creating and adding the console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_format)
    logger.addHandler(console_handler)

    return logger
