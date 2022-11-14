import types
import unittest

from steal.datasets import lasa


class TestDataset(unittest.TestCase):
    def test_lasa(self):
        """
        Test if the LASA dataset is correctly loading.
        """
        self.assertIsInstance(lasa, types.ModuleType)
