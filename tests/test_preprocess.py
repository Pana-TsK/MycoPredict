import os
import sys
# Ensure src/ is available when running this file individually
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import unittest
import numpy as np

from preprocess import Preprocess

class TestPreprocess(unittest.TestCase):
    def setUp(self):
        """
        Set up the test environment.
        """
        self.valid_smiles = ["CCO", "CCN", "CCOCC", "C1=CC=CC=C1"]
        self.invalid_smiles = ["INVALID", "C1CCCC1", "N/A"]
        self.preprocessor = Preprocess()
    
    def test_convert_to_mol_valid(self):
        """
        Test that valid SMILES strings are converted to RDKit molecules.
        """
        mols = self.preprocessor.convert_to_mol(self.valid_smiles)

        self.assertEqual(len(mols), len(self.valid_smiles), f"mols should match valid_smiles input size")  # Should match input size
        self.assertTrue(all(m is not None for m in mols), f"All molecules should be valid")  # All should be valid molecules
    
    def test_convert_to_mol_invalid(self):
        """ 
        Test that invalid SMILES are handled gracefully.
        """
        self.assertRaises(Exception, self.preprocessor.convert_to_mol, self.invalid_smiles)
    
    def test_create_xds(self):
        """
        Test that the xds are created correctly.
        """
        mols = self.preprocessor.convert_to_mol(self.valid_smiles)
        xds = self.preprocessor.create_xds(mols)

        self.assertEqual(len(xds), len(mols), f"xds should match the number of molecules")
    
    def test_create_datapoints(self):
        """
        Test that the datapoints are created correctly.
        """
        mols = self.preprocessor.convert_to_mol(self.valid_smiles)
        xds = self.preprocessor.create_xds(mols)
        datapoints = self.preprocessor.create_datapoints(self.valid_smiles, xds)

        self.assertEqual(len(datapoints), len(mols), f"datapoints should match the number of molecules")

    def test_xd_structure(self):
        """Test that features have expected shape/dtype."""
        mols = self.preprocessor.convert_to_mol(self.valid_smiles[:1])  # Single molecule
        xds = self.preprocessor.create_xds(mols)
        self.assertIsInstance(xds[0], np.ndarray)  # Check type
        self.assertTrue(xds[0].any())  # Check non-zero features


if __name__ == "__main__":
    unittest.main()