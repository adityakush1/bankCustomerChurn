"""
constants.py

this module provides constants to be used in churn_library and testing

"""

# log level
LOG_LEVEL = 'DEBUG'
TEST_LOG_LEVEL = 'DEBUG'

# log path
LOG_FILE_PATH = r'./logs/churn_library.log'
TEST_LOG_FILE_PATH = r'./logs/test_churn_library.log'

# log format
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s - %(funcName)s - %(message)s'
TEST_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(filename)s - %(message)s'

# data file
BANK_DATA_CSV_FILE = r'./data/bank_data.csv'

# directories
EDA_IMAGE_PATH: str = r'./images/eda'
SCORES_IMAGE_PATH: str = r'./images/scores'
RESULTS_IMAGE_PATH = r'./images/results'

# model specifics
LOGISTIC_REGRESSION_SOLVER = 'lbfgs'
MODELS_PATH = r'./models'
