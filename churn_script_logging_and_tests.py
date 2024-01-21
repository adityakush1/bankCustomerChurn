# library doc string
"""
churn_script_logging_and_tests.py

this module provides methods to test load dataset, perform preprocessing on the data,
do exploratory data analysis and then carry out model training methods from churn_library.py

Functions:
test_import: test for load data from file to dataframe
cat_columns : fixture for categorical columns
test_perform_eda: test perform exploratory data analysis
test_encoder_helper: testing encode categorical features
test_perform_feature_engineering: test perform feature engineering
test_train_models: test train the model
"""

import os
import logging
import pytest

from churn_library import import_data, perform_eda, perform_feature_engineering, \
    encoder_helper, train_models
import constants
from conftest import ValueStorage
from churn_logger import setup_test_logging

# Initializing the logger
logger = setup_test_logging(__name__)


def test_import():
    """
    test data import - this example is completed for you to assist with the other test functions
    """
    logging.debug("Testing import_data")
    try:
        ValueStorage.imported_df = import_data(constants.BANK_DATA_CSV_FILE)

        shape = ValueStorage.imported_df.shape
        logger.debug("Shape of imported dataframe : %s", shape)
    except FileNotFoundError as err:
        logger.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert ValueStorage.imported_df.shape[0] > 0
        assert ValueStorage.imported_df.shape[1] > 0
        logger.debug("Testing import_data: SUCCESS\n")
    except AssertionError as err:
        logger.error("Testing import_data: The file doesn't appear \
        to have rows and columns \n %s", err)
        raise err


def test_perform_eda():
    """
    test perform eda function
    """
    logger.debug("Testing perform_eda: SUCCESS \n")
    try:
        ValueStorage.eda_df = perform_eda(ValueStorage.imported_df)
        logger.debug("Testing perform_eda: %s", ValueStorage.eda_df.head(5))
    except Exception as err:
        logger.error("Testing perform_eda failed: \n %s", err)
        raise err

    try:
        assert ValueStorage.eda_df.shape[0] > 0
        assert ValueStorage.eda_df.shape[1] > 0
        logger.debug("Testing perform_eda: %s", ValueStorage.eda_df.shape)
        logger.debug("Testing perform_eda: SUCCESS\n")
    except AssertionError as err:
        logger.error("Testing perform_eda failed: \n %s", err)
        raise err


@pytest.fixture
def cat_columns():
    """
    Fixture to remove 'Attrition_Flag' column from categorical list
    """
    df_cat_cols = list(
        ValueStorage.eda_df.select_dtypes(
            include=['object']).columns)
    df_cat_cols.remove('Attrition_Flag')
    return df_cat_cols


def test_encoder_helper(cat_columns):
    """
    test encoder helper
    """
    try:
        # get churn from attrition flag - dependent variable
        logger.debug(
            "Testing test_encoder_helper: encoding 0/1 for Attrition_Flag in col - Churn")
        ValueStorage.eda_df['Churn'] = ValueStorage.eda_df['Attrition_Flag'].apply(
            lambda val: 0 if val == "Existing Customer" else 1)

        # drop Attrition_Flag column from df
        logger.debug(
            "Testing test_encoder_helper: dropping column 'Attrition_Flag'")
        df = ValueStorage.eda_df.drop('Attrition_Flag', axis=1)

        ValueStorage.encoded_df = encoder_helper(df, cat_columns, 'Churn')
        encoded_cols = ValueStorage.encoded_df.columns
        logger.debug("Testing test_encoder_helper: columns after encoding %s",
                     encoded_cols)
    except Exception as err:
        logger.error("Testing test_encoder_helper failed. \n %s", err)
        raise err

    try:
        assert "Gender_Churn" in ValueStorage.encoded_df.columns
        assert "Education_Level_Churn" in ValueStorage.encoded_df.columns
        assert "Marital_Status_Churn" in ValueStorage.encoded_df.columns
        assert "Income_Category_Churn" in ValueStorage.encoded_df.columns
        assert "Card_Category_Churn" in ValueStorage.encoded_df.columns
        assert "Attrition_Flag" not in ValueStorage.encoded_df.columns
        logger.debug("Testing test_encoder_helper: SUCCESS\n")
    except AssertionError as err:
        logger.error("Testing test_encoder_helper: Failed \n %s", err)
        raise err


def test_perform_feature_engineering():
    """
    test perform_feature_engineering
    """
    logger.debug("Testing test_perform_feature_engineering:")
    try:
        X_train, X_test, y_train, y_test = perform_feature_engineering(
            ValueStorage.eda_df)
        logger.debug("shape of X_train %s", X_train.shape)
        logger.debug("shape of X_test %s", X_test.shape)
        logger.debug("shape of y_train %s", y_train.shape)
        logger.debug("shape of y_test %s", y_test.shape)

        ValueStorage.X_train = X_train
        ValueStorage.X_test = X_test
        ValueStorage.y_train = y_train
        ValueStorage.y_test = y_test

    except Exception as err:
        logger.error("test_perform_feature_engineering : Failed \n %s", err)
        raise err

    try:
        assert ValueStorage.X_train.shape[0] > 0
        assert ValueStorage.X_train.shape[1] > 0
        assert ValueStorage.X_test.shape[0] > 0
        assert ValueStorage.X_test.shape[1] > 0
        assert ValueStorage.y_train.shape[0] > 0
        assert ValueStorage.y_test.shape[0] > 0
        logger.debug("Testing test_perform_feature_engineering: SUCCESS\n")
    except AssertionError as err:
        logger.error("test_perform_feature_engineering : Failed \n %s", err)
        raise err


def test_train_models():
    """
    test train_models
    """
    logger.debug("Testing test_train_models:")
    try:
        train_models(ValueStorage.X_train, ValueStorage.X_test,
                     ValueStorage.y_train, ValueStorage.y_test)
        model_list = os.listdir(constants.MODELS_PATH)
        logger.debug("saved models : %s", model_list)

        ValueStorage.model_list = model_list

        score_list = os.listdir(constants.SCORES_IMAGE_PATH)
        logger.debug("saved SCORES : %s", score_list)

        ValueStorage.score_list = score_list
    except Exception as err:
        logger.error("Testing test_train_models failed. \n %s", err)
        raise err

    try:
        assert ValueStorage.model_list is not None
        assert ValueStorage.model_list[0] != ""
        assert len(ValueStorage.model_list) > 0

        assert ValueStorage.score_list is not None
        assert ValueStorage.score_list[0] != ""
        assert len(ValueStorage.score_list) > 0
        logger.debug("Testing test_train_models: SUCCESS\n")
    except AssertionError as err:
        logger.error("test_train_models : Failed \n %s", err)
        raise err


if __name__ == "__main__":
    pass
