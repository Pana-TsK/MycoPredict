import pandas as pd
import numpy as np


class DataLoader:
    """
    Class to load smiles from CSV file.
    """
    def __init__(self, data_path: str, smi_col_name: str = 'smiles'):
        """
        Initialize the DataLoader class.

        Args:
            data_path (str): Path to the CSV file.
            smi_col_name (str): Column name containing SMILES strings.
        """
        self.data_path = data_path
        self.smi_col_name = smi_col_name
        self.data: pd.DataFrame = None
        self.smiles: np.ndarray = None  # Store as NumPy array

    def __str__(self) -> str:
        """
        Return a summary of the loaded data.
        """
        if self.smiles is not None:
            return f"Loaded {self.smiles.shape[0]} SMILES from {self.data_path}"
        return "No data loaded."

    def read_csv(self) -> pd.DataFrame:
        """
        Read a CSV file and store it as a DataFrame.
        """
        try:
            self.data = pd.read_csv(self.data_path)
        except Exception as e:
            raise ValueError(f"Error reading CSV file: {e}")

        return self.data

    def get_smiles(self) -> np.ndarray:
        """
        Extract the SMILES column from the DataFrame and convert it to a NumPy array.
        """
        if self.data is None:
            self.read_csv()

        if self.smi_col_name not in self.data.columns:
            raise ValueError(f"Column '{self.smi_col_name}' not found in DataFrame.")

        self.smiles = np.array(self.data[self.smi_col_name].dropna().astype(str))

        return self.smiles

    def __call__(self) -> np.ndarray:
        """
        Run the loading and extraction pipeline when the instance is called.
        """
        return self.get_smiles()

if __name__ == "__main__":
    # Example usage
    data_loader = DataLoader(data_path='src/test_file.csv')
    smiles_array = data_loader()
    print(data_loader)
    print(smiles_array)
