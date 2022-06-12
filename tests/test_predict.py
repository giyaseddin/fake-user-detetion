import os
import shutil
import unittest

import numpy as np
import pandas as pd
from pandas._testing import assert_frame_equal
from sklearn.preprocessing import LabelEncoder

from predict import label_encode_features, extract_and_save_results


class TestPredict(unittest.TestCase):
    def test_label_encode_data(self):
        df = pd.DataFrame({
            "Event": ["event1", "event1", "event2", "event3"],
            "Category": ["cat1", "cat2", "cat3", "cat4"],
        })
        event_le = LabelEncoder()
        cat_le = LabelEncoder()
        event_le.classes_ = np.array(["event1", "event2", "event3"])
        cat_le.classes_ = np.array(["cat1", "cat2", "cat3", "cat4"])

        df = label_encode_features(df, {
            "event_le": event_le, "cat_le": cat_le
        })

        assert_frame_equal(df, pd.DataFrame({
            "Event": [0, 0, 1, 2],
            "Category": [0, 1, 2, 3],
        }))

    def test_extract_and_save_results(self):
        dirname = "test_result_dir"
        os.mkdir(dirname)

        df = pd.DataFrame({
            "UserId": ["t5466", "t5467", "t5468", "t5469"],
            "Fake": [0, 0, 1, 0],
            "is_fake_probability": [0.2, 0.6, 0.7, 0.4],
        })
        m_enc = {"model_base_dir": dirname}
        output_path, report_path = extract_and_save_results(df, m_enc, dirname, by_user=True)

        assert str(os.path.basename(output_path)).endswith(".csv")
        assert str(os.path.basename(report_path)).endswith(".csv")
        assert os.path.getsize(output_path) > 0
        assert os.path.getsize(report_path) > 0

        shutil.rmtree(dirname)


if __name__ == '__main__':
    unittest.main()
