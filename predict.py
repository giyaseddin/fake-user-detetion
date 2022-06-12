import argparse

import os
from typing import Tuple

import pandas as pd
from sklearn.metrics import classification_report
from sklearn.preprocessing import OneHotEncoder

from utils import load_model_and_encoders, load_test_data, CONFIG

# Instantiate the parser
parser = argparse.ArgumentParser(
    description='This script is used for make bulk predictions of fake users detection in inference time.',
)


def label_encode_features(
        df: pd.DataFrame, m_enc: dict
) -> pd.DataFrame:
    """
    This function only transforms the categorical features 'Event' and 'Category' in a dataframe
    :param df: Test dataframe
    :param m_enc: Dictionary of label encoders that must contain 'event_le' and 'cat_le'
    :return: The same dataframe with transformed Event and Category
    """
    df[CONFIG["COLUMNS"]["Event"]] = m_enc["event_le"].transform(df[CONFIG["COLUMNS"]["Event"]])
    df[CONFIG["COLUMNS"]["Category"]] = m_enc["cat_le"].transform(df[CONFIG["COLUMNS"]["Category"]])

    return df


def predict_by_user(
        test_df: pd.DataFrame, model, ohe: OneHotEncoder
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function is used to get the predictions for the distinct set of users in the in a test set
    :param test_df: Test dataframe
    :param model: The classifier object
    :param ohe: One Hot Encoder, previously fit in the training phase
    :return: The dataframe of users with their fake probabilities, together with the granular predictions
    """
    per_user_pred = test_df.groupby(CONFIG["COLUMNS"]["UserId"]).first()

    per_user_pred[CONFIG["COLUMNS"]["Pred"]] = None
    test_df[CONFIG["COLUMNS"]["Pred"]] = None

    for user, group in test_df.groupby("UserId"):
        preds = model.predict_proba(
            ohe.transform(group[[CONFIG["COLUMNS"]["Event"], CONFIG["COLUMNS"]["Category"]]])
        )

        # Only fake class probability is counted
        fake_preds = preds[:, 1]

        per_user_pred.loc[user, CONFIG["COLUMNS"]["Pred"]] = (
                sum(fake_preds) / preds.shape[0]
        )

        for idx, pred in zip(group.index, fake_preds):
            test_df.loc[idx, CONFIG["COLUMNS"]["Pred"]] = pred

    return per_user_pred.reset_index().rename(columns={"index": "UserId"}), test_df


def extract_and_save_results(
        pred_df: pd.DataFrame, m_enc: dict, result_folder: str, by_user: bool = False, threshold: float = 0.5
) -> Tuple[str, str]:
    """
    Calculates the classification report, saves it and returns paths of the saved files.
    :param pred_df: Dataframe of the transactions and their predictions
    :param m_enc: Dictionary that holds the model directory and the label encoders
    :param result_folder: Under which the results will be saved
    :param by_user: Flag, to determine between the output format
    :param threshold: Classification threshold
    :return: Paths of the saved output and the classification report files
    """
    clf_report = classification_report(
        pred_df[CONFIG["COLUMNS"]["Fake"]],
        pred_df[CONFIG["COLUMNS"]["Pred"]].apply(lambda x: 1 if x > threshold else 0),
        output_dict=True,
        digits=4,
    )

    base_path = os.path.join(m_enc["model_base_dir"], result_folder)
    os.mkdir(base_path) if not os.path.isdir(base_path) else None

    clf_report_path = os.path.join(base_path, f"{('per_user_' if by_user else '')}clf_report.csv")
    output_path = os.path.join(base_path, f"{('per_user_' if by_user else '')}output.csv")

    pd.DataFrame(clf_report).transpose().to_csv(clf_report_path, index=True)
    if by_user:
        # Only desired columns for the output
        output_df = pred_df[
            [CONFIG["COLUMNS"]["UserId"], CONFIG["COLUMNS"]["Fake"], CONFIG["COLUMNS"]["Pred"]]
        ]
    else:
        output_df = pred_df
        output_df[CONFIG["COLUMNS"]["Event"]] = m_enc["event_le"].inverse_transform(
            output_df[CONFIG["COLUMNS"]["Event"]]
        )
        output_df[CONFIG["COLUMNS"]["Category"]] = m_enc["cat_le"].inverse_transform(
            output_df[CONFIG["COLUMNS"]["Category"]]
        )

    output_df.to_csv(output_path, index=False, float_format='%.4f')

    return output_path, clf_report_path


def start_prediction(params) -> None:
    """
    Starts the prediction procedure
    :param params: Parameters passed when calling the script
    """
    m_enc = load_model_and_encoders(params.model_path)

    test_df = load_test_data(params.test_file)

    test_df = label_encode_features(test_df, m_enc)

    per_user_pred, test_df = predict_by_user(test_df, m_enc["model"], m_enc["ohe"])

    extract_and_save_results(test_df, m_enc, params.result_folder)
    output_path, report_path = extract_and_save_results(
        per_user_pred, m_enc, params.result_folder, by_user=True
    )

    print(f"Final per-user output path: {os.path.abspath(output_path)}")


def validate_args(input_args) -> object:
    """
    This function validates the arguments passed to the script
    :param input_args: Parser arguments
    :return: Validated parser arguments
    """
    if not input_args.test_file:
        parser.error("test_file must be provided to make predictions.")

    if not input_args.result_folder:
        parser.error("result_folder must be provided to output classification report.")

    return input_args


if __name__ == '__main__':
    parser.add_argument(
        '--test_file', type=str,
        help='Required arguments, should be a path to CSV file that contains test data'
             'with the following columns: [UserId, Event, Category]',
    )
    parser.add_argument(
        '--result_folder', type=str, default=CONFIG["TEST_RESULTS_PATH"],
        help='The folder path under which the script will output classification reports.',
    )
    parser.add_argument(
        '--model_path', type=str, default=None,
        help='The path of the model to be used in the prediction.',
    )

    args = validate_args(parser.parse_args())

    start_prediction(args)
