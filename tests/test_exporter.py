import unittest

import pandas as pd
import numpy as np

import os
import sys
# Ensure src/ is available when running this file individually
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from exporter import ResultsExporter

class TestResultsExporter(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment.
        """
        self.smiles = ["CCO", "CCN", "CCOCC", "C1=CC=CC=C1"]
        self.results = np.array([0.77, 0.15, 0.99, 0.12])
        self.exporter = ResultsExporter(self.smiles, self.results)
        self.df = self.exporter.create_dataframe(self.smiles, self.results, reverse=True)
    
    def test_initialization(self):
        """
        Test that the ResultsExporter initializes correctly.
        """
        self.assertIsInstance(self.exporter, ResultsExporter, "Exporter should be an instance of ResultsExporter")
        self.assertEqual(self.exporter.smiles, self.smiles, "Smiles should be initialized correctly")
        self.assertTrue(np.array_equal(self.exporter.results, self.results), "Results should be initialized correctly")
        self.assertIsNone(self.exporter.output_path, "Output path should be None by default")
    
    def test_create_dataframe(self):
        """
        Test the create_dataframe method.
        """
        df = self.exporter.create_dataframe(self.smiles, self.results, reverse=True)
        
        # Check if the dataframe is created correctly
        self.assertIsInstance(df, pd.DataFrame, "Dataframe should be a pandas DataFrame")
        self.assertEqual(len(df), len(self.smiles), "Dataframe length should match number of smiles")

    
    def test_call(self):
        """
        Test the __call__ method.
        """
        df = self.exporter()
        
        # Check if the dataframe is created correctly
        self.assertIsInstance(df, pd.DataFrame, "Dataframe should be a pandas DataFrame")
        self.assertEqual(len(df), len(self.smiles), "Dataframe length should match number of smiles")
        

if __name__ == "__main__":
    unittest.main()