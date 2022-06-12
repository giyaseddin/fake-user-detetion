import argparse
import json
import os
import time
from typing import Tuple

import pandas as pd
from numpy import ndarray
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.model_selection._search import BaseSearchCV

from utils import (
    CONFIG,
    load_training_data,
    fit_and_save_label_encoders,
    fit_and_save_ohe,
    save_best_model,
    undersample_fake_label,
)

# Instantiate the parser
parser = argparse.ArgumentParser(
    description='This script is used for training the model using a span of training data (CSV) '
                'for Fake user detection',
)


def build_classifier(params) -> GridSearchCV:
    """
    Prepares the model for training.
    :param params: Passed when calling the script
    :return: Grid search cross validation object
    """
    if CONFIG["CLASSIFIER"] == "RF":
        model = RandomForestClassifier(random_state=0)
        p_grid = {
            'n_estimators': [1, 2, 5, ],
            'max_depth': [2, 3, 5, 10, 20],
            'criterion': ["gini", "entropy"],
            'class_weight': ["balanced", None],
        }

    elif CONFIG["CLASSIFIER"] == "LR":
        model = LogisticRegression()
        p_grid = {
            'C': [1, 2, 5, 7, 10, 20, 50],
            'max_iter': [20, 50, 100, 200, 500],
            'class_weight': ["balanced", None],
        }

    else:
        raise NotImplementedError

    if params.cross_validate:
        gkf = GroupKFold(n_splits=int(params.n_k_fold))
        gscv = GridSearchCV(estimator=model, param_grid=p_grid, cv=gkf, scoring='roc_auc')
    else:
        gscv = GridSearchCV(estimator=model, param_grid=p_grid, scoring='roc_auc')

    return gscv


def prepare_training_data(
        df: pd.DataFrame, model_folder: str, run_name: str = None
) -> Tuple[ndarray, pd.Series, ndarray]:
    """
    This function prepares the training dataset to be used in training the model.
    :param df: Training dataframe
    :param model_folder: path to save the model to
    :param run_name: Run name, unique to this training run
    :return: Tuple of: training features, target column, and the user ids. User ids only used for fair splitting
    """
    df = undersample_fake_label(df)

    event_le, cat_le, user_le = fit_and_save_label_encoders(df, model_folder, run_name=run_name)

    user_ids = user_le.transform(df[CONFIG["COLUMNS"]["UserId"]])
    df[CONFIG["COLUMNS"]["Event"]] = event_le.transform(df[CONFIG["COLUMNS"]["Event"]])
    df[CONFIG["COLUMNS"]["Category"]] = cat_le.transform(df[CONFIG["COLUMNS"]["Category"]])

    X = df[[CONFIG["COLUMNS"]["Event"], CONFIG["COLUMNS"]["Category"]]]
    ohe = fit_and_save_ohe(X, model_folder, run_name=run_name)
    X = ohe.transform(X)

    return X, df[CONFIG["COLUMNS"]["Fake"]], user_ids


def save_training_results(
        model: BaseSearchCV, model_folder: str, run_name: str = None
) -> dict:
    """
    This function saves the scores and params obtained from the training
    :param model: Cross validation search object
    :param model_folder: The folder to save the model results to
    :param run_name: Unique run name
    :return: A dictionary of best scores and best parameters
    """
    result_output = {
        "Best score": model.best_score_,
        "Best Params": model.best_params_,
    }

    base_path = os.path.join(model_folder, run_name, "train_results")
    os.mkdir(base_path) if not os.path.isdir(base_path) else None

    with open(os.path.join(base_path, "best_model_summary.json"), 'w') as f:
        json.dump(result_output, f)

    with open(os.path.join(base_path, "cv_results.txt"), 'w') as f:
        f.write(str(model.cv_results_))

    return result_output


def start_training(params) -> None:
    """
    Starts the training process
    :param params: Arguments passed while calling the script
    """
    run_name = CONFIG["RUN_NAME"] if "RUN_NAME" in CONFIG else time.strftime("%Y.%m.%d %H.%M.%S")

    train_df = load_training_data(params.training_file)

    X, y, groups = prepare_training_data(train_df, params.model_folder, run_name=run_name)

    clf = build_classifier(params)

    clf.fit(X, y, groups=groups)

    save_best_model(clf, params.model_folder, run_name=run_name)

    summary = save_training_results(clf, params.model_folder, run_name=run_name)

    print(summary)


def validate_args(input_args) -> object:
    """
    Validates the arguments passed to the script
    :param input_args:
    :return:
    """
    if not input_args.training_file:
        parser.error("training_file must be provided for training")

    if input_args.cross_validate:
        if not input_args.n_k_fold:
            parser.error("n_k_fold must be provided for cross validation")
    else:
        if hasattr(input_args, 'validate_split_ratio'):
            parser.error("validate_split_ratio shouldn't be passed if there is no 'cross_validate' flag is not used")

    if not input_args.model_folder:
        parser.error("model_folder must be provided to save output the model")

    return input_args


if __name__ == '__main__':
    parser.add_argument(
        '--training_file', type=str,
        help='Required arguments, should be a path to CSV file that contains [UserId, Event, Category, Fake] columns',
    )
    parser.add_argument(
        '--cross_validate', action='store_true',
        help='A boolean switch that determines whether to use cross validation for the training.'
             'If it\'s set, then "n_k_fold" parameter should be passed.',
    )
    parser.add_argument(
        '--n_k_fold', type=float, default=0.1,
        help='Split ratio of the training data that will be used in cross validation',
    )
    parser.add_argument(
        '--model_folder', type=str, default=CONFIG["MODEL_PATH"],
        help='The folder path under which the script will output the model.',
    )

    args = validate_args(parser.parse_args())

    start_training(args)
