import os
import json
import pickle
from typing import Tuple

import numpy as np
import pandas as pd
from numpy import ndarray
from sklearn.model_selection._search import BaseSearchCV
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

CONFIG = json.load(open(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "config.json")
))


def load_training_data(path: str) -> pd.DataFrame:
    """
    Simply loads the training data
    :param path: Path of the training data file
    :return: Dataframe of the training data
    """
    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df.drop(columns="Unnamed: 0", inplace=True)

    return df


def load_test_data(path: str) -> pd.DataFrame:
    """
    Simply loads the test set
    :param path: Path of the test set
    :return: Dataframe of the test data
    """
    df = pd.read_csv(path)

    return df


def undersample_fake_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    This function ensures the balance between the two classes
    The imbalance is treated with regard to the transactions not users, although final classification is done to
    users, better results observed with transaction-based undersampling.
    :param df: Dataframe we want to undergo undersampling
    :return: New version of the dataframe, but with balanced (reduced) negative class
    """
    df_fake = df[df[CONFIG["COLUMNS"]["Fake"]] == 1]
    df_real_sampled = df[df[CONFIG["COLUMNS"]["Fake"]] == 0].sample(df_fake.shape[0], random_state=0)

    return pd.concat([df_fake, df_real_sampled], axis=0).sample(frac=1., random_state=0)


def fit_and_save_label_encoders(
        df: pd.DataFrame, model_folder: str, run_name: str = None
) -> Tuple[LabelEncoder, LabelEncoder, LabelEncoder]:
    """
    In the function, the label encoders for the categorical features are fitted and saved
    :param df: Dataframe of the training set
    :param model_folder: Folder path, under which the fitted encoders will be saved
    :param run_name: Unique to the current training run
    :return: Fitted label encoder objects for: event, category and user. User is not significant, but used
     because it's useful for user-based grouping
    """
    event_le = LabelEncoder()
    cat_le = LabelEncoder()
    user_le = LabelEncoder()

    event_le.fit(df[CONFIG["COLUMNS"]["Event"]])
    cat_le.fit(df[CONFIG["COLUMNS"]["Category"]])
    user_le.fit(df[CONFIG["COLUMNS"]["UserId"]])  # Only used for splitting folds as groups

    base_path = os.path.join(model_folder, run_name, "model")
    os.mkdir(os.path.dirname(base_path)) if not os.path.isdir(os.path.dirname(base_path)) else None
    os.mkdir(base_path) if not os.path.isdir(base_path) else None
    np.save(os.path.join(base_path, "event_le.npy"), event_le.classes_)
    np.save(os.path.join(base_path, "cat_le.npy"), cat_le.classes_)
    np.save(os.path.join(base_path, "user_le.npy"), user_le.classes_)

    return event_le, cat_le, user_le


def fit_and_save_ohe(
        X: ndarray, model_folder: str, run_name: str = None
) -> OneHotEncoder:
    """
    Obviously, this function fits a one hot encoder, and saves it :)
    :param X: Extracted features of the training set
    :param model_folder: folder to save the fitted encoder
    :param run_name: Unique run name
    :return: Fitted One hot encoder
    """
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X)

    base_path = os.path.join(model_folder, run_name, "model")
    with open(os.path.join(base_path, "ohe.pkl"), "wb") as f:
        pickle.dump(ohe, f)

    return ohe


def save_best_model(
        model: BaseSearchCV, model_folder: str, run_name: str = None
) -> None:
    """
    This function saves the best model after the cross validation
    :param model: Search cross validation object
    :param model_folder: path where the model will be saved
    :param run_name: Unique run name
    """
    base_path = os.path.join(model_folder, run_name, "model")
    with open(os.path.join(base_path, "model.pkl"), 'wb') as f:
        pickle.dump(model.best_estimator_, f)


def check_model_path(model_path: str) -> str:
    """
    Simple validation to the existence of the model under the given (of default) model path
    :param model_path: Model folder path
    :return: Checked model folder
    """
    base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), CONFIG["MODEL_PATH"])
    if not model_path:
        model_path = os.path.join(base_dir, max(
            [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
        ))

    if (
            not os.path.isdir(model_path) or
            not os.path.isdir(os.path.join(model_path, "model")) or
            not os.path.isfile(os.path.join(model_path, "model", "model.pkl"))
    ):
        raise FileNotFoundError(f"There is no model in the model path '{model_path}'")

    model_path = os.path.join(model_path, "model") if os.path.basename(model_path) != "model" else model_path

    return model_path


def load_model_and_encoders(model_path: str) -> dict:
    """
    This function loads the saved model and the label and one hot encoders
    :param model_path: Model folder path
    :return: Dictionary of the model base directory, the mode, and the encoders of the features
    """
    model_path = check_model_path(model_path)

    model = pickle.load(open(os.path.join(model_path, "model.pkl"), 'rb'))
    ohe = pickle.load(open(os.path.join(model_path, "ohe.pkl"), 'rb'))

    cat_le = LabelEncoder()
    event_le = LabelEncoder()
    cat_le.classes_ = np.load(os.path.join(model_path, "cat_le.npy"), allow_pickle=True)
    event_le.classes_ = np.load(os.path.join(model_path, "event_le.npy"), allow_pickle=True)

    return {
        "model_base_dir": os.path.dirname(model_path),
        "model": model,
        "ohe": ohe,
        "cat_le": cat_le,
        "event_le": event_le,
    }
