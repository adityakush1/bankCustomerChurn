# library doc string
"""
churn_library.py

this module provides methods to load dataset, perform preprocessing on the data,
do exploratory data analysis and then carry out model training

Functions:
general_logger: general logging for entry /exit
import_data: load data from file to dataframe
perform_eda: perform exploratory data analysis
encoder_helper: encode categorical features
perform_feature_engineering: perform feature engineering
classification_report_image: generate classification report image
feature_importance_plot: fetch important features from dataset
train_model: train the model
"""

# import libraries

import os

import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report

import constants
from churn_logger import setup_logging



os.environ['QT_QPA_PLATFORM'] = 'offscreen'

sns.set()


# Initializing the logger
logger = setup_logging(__name__)


def general_logger(func):
    """
    Decorator that adds entry and exit logging to function calls
    """
    def inner(*args, **kwargs):
        logger.debug("%s - Enter *********", func.__name__)
        print(*args)
        result = func(*args, **kwargs)
        logger.debug("%s - Exit ********\n\n", func.__name__)
        return result
    return inner


@general_logger
def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    try:
        df = pd.read_csv(pth)
        logger.debug('File %s loaded successfully to dataframe ', pth)
        logger.debug('Shape of DataFrame from file : %s', df.shape)
    except FileNotFoundError as err:
        logger.debug('Failed to load file %s . Exception : %s ', pth, err)
        raise err

    return df


@general_logger
def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            df
    """
    logger.debug("Shape of dataframe : %s", df.shape)

    # check for missing values, if found drop
    na_check = (df.isnull().sum() > 0).any()
    if na_check:
        df.dropna(inplace=True)

    # describe df
    logger.debug("describing Dataframe : %s", df.describe())

    # to see distribution of attrition
    plt.figure(figsize=(10, 5))
    df['Attrition_Flag'].value_counts().plot(kind='bar')
    plt.ylabel('Count')
    plt.xlabel('Attrition_Flag')
    plt.savefig(
        constants.EDA_IMAGE_PATH +
        '/attrition_flag_hist.png',
        bbox_inches='tight')
    plt.close()

    # customer age distribution
    plt.figure(figsize=(20, 10))
    df['Customer_Age'].hist()
    plt.ylabel('Count')
    plt.xlabel('Customer_Age')
    plt.savefig(
        constants.EDA_IMAGE_PATH +
        '/Customer_Age_hist.png',
        bbox_inches='tight')
    plt.close()

    # count of customers and their marital status
    plt.figure(figsize=(10, 5))
    df.Marital_Status.value_counts().plot(kind='bar')
    plt.savefig(
        constants.EDA_IMAGE_PATH +
        '/Marital_Status_hist.png',
        bbox_inches='tight')
    plt.close()

    # distribution of Total_Trans_Ct
    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)

    plt.title('Histogram with KDE for Total Transactions Count')
    plt.savefig(
        constants.EDA_IMAGE_PATH +
        '/Total_Trans_Ct_histplot.png',
        bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(20, 15))
    sns.heatmap(df.corr(), annot=False, cmap='coolwarm', linewidths=2)
    plt.title('Correlation matrix between features')
    plt.savefig(
        constants.EDA_IMAGE_PATH +
        '/Correlation_heatmap.png',
        bbox_inches='tight')
    plt.close()

    return df


@general_logger
def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for \
            naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """

    logger.debug("Got the category list as %s", category_lst)

    # generalised encoding logic
    for col_item in category_lst:
        col_lst = []
        col_groups = df.groupby(col_item).mean()[response]

        for val in df[col_item]:
            col_lst.append(col_groups.loc[val])

        logger.debug('here....' + col_item + '_' + response)
        df[col_item + '_' + response] = col_lst

    return df


@general_logger
def perform_feature_engineering(df):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming
              variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """

    df_cat_cols = list(df.select_dtypes(include=['object']).columns)
    logger.debug("Categorical Columns \n %s", df_cat_cols)

    # df_num_cols = [
    #     'Customer_Age',
    #     'Dependent_count',
    #     'Months_on_book',
    #     'Total_Relationship_Count',
    #     'Months_Inactive_12_mon',
    #     'Contacts_Count_12_mon',
    #     'Credit_Limit',
    #     'Total_Revolving_Bal',
    #     'Avg_Open_To_Buy',
    #     'Total_Amt_Chng_Q4_Q1',
    #     'Total_Trans_Amt',
    #     'Total_Trans_Ct',
    #     'Total_Ct_Chng_Q4_Q1',
    #     'Avg_Utilization_Ratio'
    # ]

    # get churn from attrition flag - dependent variable
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)

    # drop Attrition_Flag column from df
    df.drop('Attrition_Flag', axis=1, inplace=True)

    df_cat_cols.remove('Attrition_Flag')
    logger.debug("Categorical Column after dropping \n %s", df_cat_cols)

    y = df['Churn']
    X = pd.DataFrame()

    df = encoder_helper(df, df_cat_cols, 'Churn')

    logger.debug(df.columns)
    keep_cols = [
        'Customer_Age',
        'Dependent_count',
        'Months_on_book',
        'Total_Relationship_Count',
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon',
        'Credit_Limit',
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy',
        'Total_Amt_Chng_Q4_Q1',
        'Total_Trans_Amt',
        'Total_Trans_Ct',
        'Total_Ct_Chng_Q4_Q1',
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn']

    X[keep_cols] = df[keep_cols]
    logger.debug(X.head(3))

    # train test split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)

    return X_train, X_test, y_train, y_test


@general_logger
def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """

    # save as figure
    plt.figure(figsize=(10, 5))
    plt.title('Test set - Classification report for Random Forest')
    sns.heatmap(pd.DataFrame(classification_report(y_test,
                                                   y_test_preds_rf,
                                                   output_dict=True)).iloc[:-1, :].T, annot=True)
    plt.savefig(
        constants.SCORES_IMAGE_PATH +
        '/CLR_RF_test.png',
        bbox_inches='tight')
    plt.close()

    # save as image file
    plt.figure(figsize=(10, 5))
    plt.title('Training set - Classification report for Random Forest')
    sns.heatmap(pd.DataFrame(classification_report(
        y_train, y_train_preds_rf, output_dict=True)).iloc[:-1, :].T, annot=True)
    plt.savefig(
        constants.SCORES_IMAGE_PATH +
        '/CLR_RF_train.png',
        bbox_inches='tight')
    plt.close()

    # save as figure
    plt.figure(figsize=(10, 5))
    plt.title('Test set - Classification report for Logistic Regression')
    sns.heatmap(pd.DataFrame(classification_report(
        y_test, y_test_preds_lr, output_dict=True)).iloc[:-1, :].T, annot=True)
    plt.savefig(
        constants.SCORES_IMAGE_PATH +
        '/CLR_LR_test.png',
        bbox_inches='tight')
    plt.close()

    # save as image file
    plt.figure(figsize=(10, 5))
    plt.title('Training set - Classification report for Logistic Regression')
    sns.heatmap(pd.DataFrame(classification_report(
        y_train, y_train_preds_lr, output_dict=True)).iloc[:-1, :].T, annot=True)
    plt.savefig(
        constants.SCORES_IMAGE_PATH +
        '/CLR_LR_train.png',
        bbox_inches='tight')
    plt.close()


@general_logger
def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    # shap summary plot
    explainer = shap.TreeExplainer(model.best_estimator_)
    shap_values = explainer.shap_values(X_data)
    shap.summary_plot(shap_values, X_data, plot_type="bar", show=False)
    plt.savefig(output_pth + '/shap_summary_plot.png', bbox_inches='tight')
    plt.close()

    # Calculate feature importances
    importances = model.best_estimator_.feature_importances_

    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(
        output_pth +
        '/feature_importance_plot.png',
        bbox_inches='tight')
    plt.close()


@general_logger
def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    logger.debug('random forest results')
    # random forest search
    rfc = RandomForestClassifier(random_state=42)

    # logistic regression
    lrc = LogisticRegression(
        solver=constants.LOGISTIC_REGRESSION_SOLVER,
        max_iter=3000)

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)

    # training model rfc
    cv_rfc.fit(X_train, y_train)
    joblib.dump(cv_rfc, constants.MODELS_PATH + '/rfc_model.pkl')

    # training model lrc
    lrc.fit(X_train, y_train)
    joblib.dump(lrc, constants.MODELS_PATH + '/lrc_model.pkl')

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    # scores
    logger.debug('random forest results')
    logger.debug('test results')
    logger.debug(classification_report(y_test, y_test_preds_rf))
    logger.debug('train results')
    logger.debug(classification_report(y_train, y_train_preds_rf))

    logger.debug('logistic regression results')
    logger.debug('test results')
    logger.debug(classification_report(y_test, y_test_preds_lr))
    logger.debug('train results')
    logger.debug(classification_report(y_train, y_train_preds_lr))

    # generate classification report
    classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf)

    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    lrc_plot.figure_.savefig(
        constants.SCORES_IMAGE_PATH +
        '/lrc_roc.png',
        bbox_inches='tight')
    plt.close()

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    rfc_disp.figure_.savefig(
        constants.SCORES_IMAGE_PATH +
        '/rfc_roc.png',
        bbox_inches='tight')
    plt.close()

    # generate feature importance
    feature_importance_plot(cv_rfc, X_test, constants.RESULTS_IMAGE_PATH)


if __name__ == '__main__':
    # 1. import data from csv to dataframe and return
    data = import_data(constants.BANK_DATA_CSV_FILE)
    logger.debug(data.head(5))

    # 2. perform_eda on dataframe data
    data = perform_eda(data)

    # 3. perform_feature_engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(data)

    # 4. perform train_models
    train_models(X_train, X_test, y_train, y_test)
