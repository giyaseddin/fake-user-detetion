import os.path
import unittest

import numpy as np
import pandas as pd

from utils import undersample_fake_label, check_model_path


class TestUtils(unittest.TestCase):
    def test_config(self):
        import utils
        assert type(utils.CONFIG) == dict
        assert "LABELS_ID2NAME" in utils.CONFIG
        assert "COLUMNS" in utils.CONFIG
        assert "CLASSIFIER" in utils.CONFIG
        assert "MODEL_PATH" in utils.CONFIG
        assert "TEST_RESULTS_PATH" in utils.CONFIG

    def test_undersample(self):
        df = pd.DataFrame({
            'ID': [1, 2, 3, 4, 5, 6, 7, 8],
            'Fake': [1, 1, 1, 0, 0, 0, 0, 0],
        })

        undersampled = undersample_fake_label(df)

        np.array_equal(
            undersampled[['Fake']].values,
            pd.DataFrame({
                'Fake': [1, 1, 1, 0, 0, 0],
            }).values,
        )

    def test_check_model_path(self):
        model_path = check_model_path(None)
        assert os.path.basename(model_path) == "model"

        with self.assertRaises(FileNotFoundError) as context:
            check_model_path("arbitrary path")


if __name__ == '__main__':
    unittest.main()
