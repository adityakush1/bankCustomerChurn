"""
churn_library.py

this module provides methods for logging purposes

Functions:
setup_logging: logging for churn_library.py
setup_test_logging: logging for churn-script_logging_and_tests.py
"""


import logging

import constants


def setup_logging(var):
    """
    returns logger for churn_library.py

    input:
            var : __name__ from the calling module
    output:
            logger: logging.getLogger
    """
    logger = logging.getLogger(var)

    if constants.LOG_LEVEL == 'DEBUG':
        log_level = logging.DEBUG
    elif constants.LOG_LEVEL == 'INFO':
        log_level = logging.INFO
    elif constants.LOG_LEVEL == 'WARNING':
        log_level = logging.WARNING
    else:
        log_level = logging.ERROR

    logger.setLevel(log_level)

    # Create a file handler
    file_handler = logging.FileHandler(constants.LOG_FILE_PATH)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the file handler
    formatter = logging.Formatter(constants.LOG_FORMAT)
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger


def setup_test_logging(var):
    """
    returns logger for churn_script_logging_and_tests.py

    input:
            var : __name__ from the calling module
    output:
            logger: logging.getLogger
    """
    logger = logging.getLogger(var)

    if constants.TEST_LOG_LEVEL == 'DEBUG':
        log_level = logging.DEBUG
    elif constants.TEST_LOG_LEVEL == 'INFO':
        log_level = logging.INFO
    elif constants.TEST_LOG_LEVEL == 'WARNING':
        log_level = logging.WARNING
    else:
        log_level = logging.ERROR

    logger.setLevel(log_level)

    # Create a file handler
    file_handler = logging.FileHandler(constants.TEST_LOG_FILE_PATH)
    file_handler.setLevel(logging.DEBUG)

    # Create a formatter and add it to the file handler
    formatter = logging.Formatter(constants.TEST_LOG_FORMAT)
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    return logger
