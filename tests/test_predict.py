import os
import sys

# Ensure src/ is available when running this file individually
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import unittest
import numpy as np

from preprocess import Preprocess
from predict import Predict


class TestPredictor(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment.
        """
        self.valid_smiles = ["CCO", "CCN", "CCOCC", "C1=CC=CC=C1"]
        self.example_datapoints = Preprocess()(self.valid_smiles)
    
    def test_predictor_initialization(self):
        """
        Test that the predictor initializes correctly.
        """
        predictor = Predict()
        self.assertIsNotNone(predictor.model, "Model should be initialized correctly")
    
    def test_build_test_dataset(self):
        """
        test whether the dataset is compiled correctly.
        """
        self.test_dset, self.test_loader = Predict().build_test_dataset(self.example_datapoints)

        # Check if the dataset and dataloader are properly initialized
        self.assertIsNotNone(self.test_dset, "Error: test_dset should not be None")
        self.assertIsNotNone(self.test_loader, "Error: test_loader should not be None")


if __name__ == "__main__":
    unittest.main()
