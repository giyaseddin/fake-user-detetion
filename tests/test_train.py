import os
import shutil
import unittest

import numpy as np
import pandas as pd
from numpy.testing import assert_array_equal
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold

from train import prepare_training_data


class TestTrain(unittest.TestCase):
    def test_build_classifier_RF_CV(self):
        import train
        train.CONFIG["CLASSIFIER"] = "RF"

        class Param:
            cross_validate = True
            n_k_fold = 4

        search_obj = train.build_classifier(Param())

        self.assertIsInstance(search_obj.cv, GroupKFold)
        self.assertIsInstance(search_obj.estimator, RandomForestClassifier)

    def test_build_classifier_RF(self):
        import train
        train.CONFIG["CLASSIFIER"] = "RF"

        class Param:
            cross_validate = False

        search_obj = train.build_classifier(Param())

        assert search_obj.cv is None
        self.assertIsInstance(search_obj.estimator, RandomForestClassifier)

    def test_build_classifier_LR_CV(self):
        import train
        train.CONFIG["CLASSIFIER"] = "LR"

        class Param:
            cross_validate = True
            n_k_fold = 4

        search_obj = train.build_classifier(Param())

        self.assertIsInstance(search_obj.cv, GroupKFold)
        self.assertIsInstance(search_obj.estimator, LogisticRegression)

    def test_build_classifier_LR(self):
        import train
        train.CONFIG["CLASSIFIER"] = "LR"

        class Param:
            cross_validate = False

        search_obj = train.build_classifier(Param())

        assert search_obj.cv is None
        self.assertIsInstance(search_obj.estimator, LogisticRegression)

    def test_build_classifier_not_implemented(self):
        import train
        train.CONFIG["CLASSIFIER"] = "different"

        class Param:
            cross_validate = False

        with self.assertRaises(NotImplementedError) as context:
            train.build_classifier(Param())

    def test_prepare_training_data(self):
        dirname = "test_model_folder"
        os.mkdir(dirname)

        df = pd.DataFrame({
            "UserId": [66, 67, 68, 69],
            "Event": [1, 2, 2, 1],
            "Category": [1, 2, 3, 4],
            "Fake": [1, 0, 0, 1],
        })

        X, y, groups = prepare_training_data(df, dirname, "test_run")

        expected_X = np.array([
            [0.00000, 1.00000, 0.00000, 0.00000, 1.00000, 0.00000],
            [0.00000, 1.00000, 0.00000, 1.00000, 0.00000, 0.00000],
            [1.00000, 0.00000, 0.00000, 0.00000, 0.00000, 1.00000],
            [1.00000, 0.00000, 1.00000, 0.00000, 0.00000, 0.00000],

        ])
        expected_y = np.array([0, 0, 1, 1])
        expected_groups = np.array([2, 1, 3, 0])

        assert_array_equal(X.toarray(), expected_X)
        assert_array_equal(y.to_numpy(), expected_y)
        assert_array_equal(groups, expected_groups)

        shutil.rmtree(dirname)


if __name__ == '__main__':
    unittest.main()
