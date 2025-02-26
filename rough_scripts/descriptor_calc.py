from rdkit import Chem
from rdkit.Chem import Descriptors, Descriptors3D
from rdkit.Chem import AllChem  # For generating 3D conformers
import pandas as pd
from typing import Optional, List

class Compound:
    def __init__(self, smiles: str, hit_miss: Optional[int] = None):
        self.smiles = smiles
        self.hit_miss = hit_miss  # 1 for hit, 0 for miss
        self.descriptors: Optional[pd.Series] = None

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
    def __init__(self, descriptor_list : list):
        self.descriptor_names = descriptor_list

    def calculate_descriptors(compound):
        """Dynamically calculates all descriptors (2D and 3D) for a given compound."""
        mol = Chem.MolFromSmiles(compound.smiles)
        if mol:
            # Calculate 2D descriptors
            descriptors_2d = {name: getattr(Descriptors, name)(mol) for name in DescriptorCalculator.descriptor_names if hasattr(Descriptors, name)}

            compound.set_descriptors(descriptors_2d)
        else:
            raise ValueError(f"Invalid SMILES: {compound.smiles}")

class Dataset:
    def __init__(self, csv_path: str, smiles_col: str, descriptor_list : list, hit_miss_col: str = None):
        """
        Initializes the dataset.
        :param csv_path: Path to the CSV file.
        :param smiles_col: Name of the column containing SMILES strings.
        :param hit_miss_col: Name of the column containing hit/miss labels.
        """
        self.csv_path = csv_path
        self.smiles_col = smiles_col
        if hit_miss_col is not None:
            self.hit_miss_col = hit_miss_col
        self.compounds: List[Compound] = []
        self.descriptor_list = descriptor_list
    
    def __repr__(self):
        return f"Dataset with {len(self.compounds)} compounds"

    def load_data(self):
        """Loads the dataset from the CSV file and creates Compound objects."""
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            smiles = row[self.smiles_col]
            if hit_miss_col is not None:
                hit_miss = row[self.hit_miss_col]
                compound = Compound(smiles, hit_miss)
            else:
                compound = Compound(smiles)
            self.compounds.append(compound)

    def calculate_descriptors(self):
        """Calculates descriptors for all compounds in the dataset."""
        for compound in self.compounds:
            DescriptorCalculator(self.descriptor_list).calculate_descriptors(compound)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the dataset into a DataFrame."""
        return pd.concat([compound.to_dataframe() for compound in self.compounds], ignore_index=True)

# Example usage
if __name__ == "__main__":
    # Path to your CSV file
    csv_path = r"C:\Users\panag\OneDrive\Documents\coding\Projects\MycoPredict\data\training_data\training_dataset\training_dataset.csv"
    smiles_col = "canonical_smiles"  # Column name for SMILES strings
    hit_miss_col = "class"  # Column name for hit/miss labels

    # Load dataset
    dataset = Dataset(csv_path, smiles_col, hit_miss_col)
    dataset.load_data()

    # Calculate descriptors
    dataset.calculate_descriptors()
    # The ability to add 3d descriptors was added, but the function is not called

    # Convert to DataFrame
    df = dataset.to_dataframe()
    print(df.head())

    # Save to descriptors file:
    df.to_csv(r"C:\Users\panag\OneDrive\Documents\coding\Projects\MycoPredict\data\training_data\descriptors\06_testdescriptors.csv", index=False)