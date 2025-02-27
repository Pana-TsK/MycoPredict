from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from typing import Optional, List

class Compound:
    """
    A class representing a single compound with its SMILES string and hit/miss label.
    The class also stores calculated descriptors for the compound.
    """
    def __init__(self, smiles: str, hit_miss: Optional[int] = None):
        self.smiles = smiles
        self.hit_miss = hit_miss  # 1 for hit, 0 for miss, remains optional
        self.descriptors: Optional[pd.Series] = None
    
    def __str__(self):
        return f"Compound with SMILES: {self.smiles}"
    
    def __repr__(self):
        return f"Compound({self.smiles})"

    def set_descriptors(self, descriptors: dict):
        """Stores descriptors as a pandas Series for easier manipulation."""
        self.descriptors = pd.Series(descriptors)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the compound data into a single-row DataFrame."""
        data = {"canonical_smiles": self.smiles}
        if self.hit_miss is not None:
            data["Hit_Miss"] = self.hit_miss  # Add hit/miss column
        if self.descriptors is not None:
            data.update(self.descriptors.to_dict())  # Merge descriptors into dict
        return pd.DataFrame([data])

class DescriptorCalculator:
    """
    Class performing RDKit descriptor calculations.
    Initialized with a list of predefined descriptor names.
    """
    def __init__(self, descriptor_list: List[str]):
        self.descriptor_names = descriptor_list

    def __repr__(self):
        return f"DescriptorCalculator({self.descriptor_names})"
    
    def __str__(self):
        return f"DescriptorCalculator with descriptors: {self.descriptor_names}"

    def calculate_descriptors(self, smiles: str) -> pd.Series:
        """Dynamically calculates all descriptors for a given compound."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {smiles}")

        descriptors = {}
        for name in self.descriptor_names:
            if hasattr(Descriptors, name):
                descriptors[name] = getattr(Descriptors, name)(mol)
            else:
                raise ValueError(f"Descriptor {name} not found")
        
        return pd.Series(descriptors)
    
class MolSet:
    """
    A manager class responsible for handling a dataset of SMILES strings and descriptors.
    
    :param input_path: Path to the CSV file containing the dataset.
    :param output_path: Path where processed data will be saved.
    :param smiles_col: Column name containing SMILES strings (default: 'canonical_smiles').
    :param hit_miss_col: Column name indicating hit/miss status (default: None).
    """
    def __init__(self, input_path: str, output_path: str, dcalc : DescriptorCalculator, smiles_col: str = 'canonical_smiles', hit_miss_col: Optional[str] = None):
        # Define the path and necessary columns
        self.output_path = output_path
        self.smiles_col = smiles_col
        self.hit_miss_col = hit_miss_col

        # Initialize the raw dataframe and the compound list
        self.compounds: List[Compound] = []  # Store compounds as objects
        self.df = self.create_dataset(input_path)  # Load dataset into a DataFrame

        # Store the descriptor calculator
    
    def __str__(self):
        return f"Dataset with {len(self.compounds)} compounds"

    def create_dataset(self, input_path):
        """Loads the dataset and creates Compound objects for each row."""
        try:
            df = pd.read_csv(input_path)
        except Exception as e:
            raise ValueError(f"Error reading input file: {e}")

        # Ensure SMILES column exists
        if self.smiles_col not in df.columns:
            raise ValueError(f"Column '{self.smiles_col}' not found in dataset")
        
        if self.hit_miss_col and self.hit_miss_col not in df.columns:
            raise ValueError(f"Column '{self.hit_miss_col}' not found in dataset")

        # Create Compound objects
        for _, row in df.iterrows():
            smiles = row[self.smiles_col]
            hit_miss = row[self.hit_miss_col] if self.hit_miss_col else None
            compound = Compound(smiles, hit_miss)
            self.compounds.append(compound)

        return df  # Store DataFrame for direct access

    def compute_descriptors(self):
        pass # Implement this

    def get_dataframe(self) -> pd.DataFrame:
        """Converts the dataset to a DataFrame."""
        df_list = [compound.to_dataframe() for compound in self.compounds]
        return pd.concat(df_list, ignore_index=True)

    def save_dataset(self, path=None):
        """Saves the dataset to a CSV file."""
        if path is None:
            path = self.output_path
        self.get_dataframe().to_csv(path, index=False)

if __name__ == "__main__":
    # Initialize descriptor calculator
    desc_list = ['MolWt', 'NumHDonors', 'NumHAcceptors', 'TPSA']
    calc = DescriptorCalculator(desc_list)

    # Initialize dataset
    input_path = r'C:\Users\panag\OneDrive\Documents\coding\Projects\MycoPredict\data\validation_data\val_dataset.csv'
    output_path = 'necessary'
    
    dataset = MolSet(input_path, output_path, calc)

    # print contents
    print(dataset)

    print(dataset.get_dataframe().head())