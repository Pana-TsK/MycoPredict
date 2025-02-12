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
        data = {"SMILES": self.smiles}
        if self.hit_miss is not None:
            data["Hit_Miss"] = self.hit_miss  # Add hit/miss column
        if self.descriptors is not None:
            data.update(self.descriptors.to_dict())  # Merge descriptors into dict
        return pd.DataFrame([data])

class DescriptorCalculator:
    descriptor_names = [
        # Global Physicochemical Properties
        "MolLogP", "TPSA", "FractionCSP3", "HeavyAtomMolWt", "LabuteASA", "HallKierAlpha",

        # Structural and Topological Descriptors
        "BertzCT", "Chi0n", "Chi4n", "NumAromaticRings", "NumAliphaticRings", 
        "NumRotatableBonds", "NumHeteroatoms",

        # Hydrogen Bonding and Polarity
        "NumHDonors", "NumHAcceptors", "EState_VSA3",

        # 3D Descriptors
        "RadiusOfGyration", "PMI1", "PMI2", "PMI3", "InertialShapeFactor", 
        "Asphericity", "SpherocityIndex",

        # Target-Specific Descriptors
        # These are currently not being calculated, but could be added in later with custom methods
        "NumHalogens", "NumElectronWithdrawingGroups", "NumElectronDonatingGroups", 
        "Redox_Potential"
    ]

    @staticmethod
    def generate_3d_conformer(mol):
        """Generates and optimizes a 3D conformer for a molecule."""
        mol = Chem.AddHs(mol)  # Add hydrogens for accurate 3D geometry
        AllChem.EmbedMolecule(mol)  # Generate 3D conformer
        AllChem.MMFFOptimizeMolecule(mol)  # Optimize geometry
        return mol

    @staticmethod
    def calculate_3d_descriptors(mol):
        """Calculates 3D descriptors for a molecule."""
        return {
            "RadiusOfGyration": Descriptors3D.RadiusOfGyration(mol),
            "PMI1": Descriptors3D.PMI1(mol),
            "PMI2": Descriptors3D.PMI2(mol),
            "PMI3": Descriptors3D.PMI3(mol),
            "InertialShapeFactor": Descriptors3D.InertialShapeFactor(mol),
            "Asphericity": Descriptors3D.Asphericity(mol),
            "SpherocityIndex": Descriptors3D.SpherocityIndex(mol),
        }

    @staticmethod
    def calculate_descriptors(compound):
        """Dynamically calculates all descriptors (2D and 3D) for a given compound."""
        mol = Chem.MolFromSmiles(compound.smiles)
        if mol:
            # Calculate 2D descriptors
            descriptors_2d = {name: getattr(Descriptors, name)(mol) for name in DescriptorCalculator.descriptor_names if hasattr(Descriptors, name)}
            
            # Generate 3D conformer and calculate 3D descriptors
            mol_3d = DescriptorCalculator.generate_3d_conformer(mol)
            descriptors_3d = DescriptorCalculator.calculate_3d_descriptors(mol_3d)
            
            # Combine 2D and 3D descriptors
            all_descriptors = {**descriptors_2d, **descriptors_3d}
            compound.set_descriptors(all_descriptors)
        else:
            raise ValueError(f"Invalid SMILES: {compound.smiles}")

class Dataset:
    def __init__(self, csv_path: str, smiles_col: str, hit_miss_col: str):
        """
        Initializes the dataset.
        :param csv_path: Path to the CSV file.
        :param smiles_col: Name of the column containing SMILES strings.
        :param hit_miss_col: Name of the column containing hit/miss labels.
        """
        self.csv_path = csv_path
        self.smiles_col = smiles_col
        self.hit_miss_col = hit_miss_col
        self.compounds: List[Compound] = []

    def load_data(self):
        """Loads the dataset from the CSV file and creates Compound objects."""
        df = pd.read_csv(self.csv_path)
        for _, row in df.iterrows():
            smiles = row[self.smiles_col]
            hit_miss = row[self.hit_miss_col]
            compound = Compound(smiles, hit_miss)
            self.compounds.append(compound)

    def calculate_descriptors(self):
        """Calculates descriptors for all compounds in the dataset."""
        for compound in self.compounds:
            DescriptorCalculator.calculate_descriptors(compound)

    def to_dataframe(self) -> pd.DataFrame:
        """Converts the dataset into a DataFrame."""
        return pd.concat([compound.to_dataframe() for compound in self.compounds], ignore_index=True)

# Example usage
if __name__ == "__main__":
    # Path to your CSV file
    csv_path = r"C:\Users\panag\OneDrive\Documents\coding\Projects\AIbiotics\mycobacteria_ml_project\training_data\training_dataset\training_dataset.csv"
    smiles_col = "canonical_smiles"  # Column name for SMILES strings
    hit_miss_col = "class"  # Column name for hit/miss labels

    # Load dataset
    dataset = Dataset(csv_path, smiles_col, hit_miss_col)
    dataset.load_data()

    # Calculate descriptors
    dataset.calculate_descriptors()

    # Convert to DataFrame
    df = dataset.to_dataframe()
    print(df.head())

    # Save to descriptors file:
    df.to_csv(r"C:\Users\panag\OneDrive\Documents\coding\Projects\AIbiotics\mycobacteria_ml_project\training_data\descriptors\05_descriptors.csv", index=False)