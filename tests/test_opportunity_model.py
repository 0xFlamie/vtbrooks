import os
import sys
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from opportunity_model import triple_barrier


class TestTripleBarrier(unittest.TestCase):
    def setUp(self):
        self.df = pd.DataFrame({"close": [100, 100, 100], "high": [100, 102, 100], "low": [100, 99.5, 98]})

    def test_long_hits_target_first(self):
        self.assertEqual(triple_barrier(self.df, 0, 1, 1, 2), 1)

    def test_short_hits_stop_first(self):
        self.assertEqual(triple_barrier(self.df, 0, -1, 1, 2), -1)

    def test_same_bar_touch_is_ambiguous(self):
        df = pd.DataFrame({"close": [100, 100], "high": [100, 101], "low": [100, 99]})
        self.assertEqual(triple_barrier(df, 0, 1, 1, 1), 0)


if __name__ == "__main__":
    unittest.main()
