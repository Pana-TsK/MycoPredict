from rdkit import Chem
from rdkit.Chem import Descriptors
import pandas as pd
from typing import Optional, List

class Compound:
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
    def __init__(self, descriptor_list : list[str]):
        self.descriptor_names = descriptor_list


    def __repr__(self):
        return f"DescriptorCalculator({self.descriptor_names})"
    
    def __str__(self):
        return f"DescriptorCalculator with descriptors: {self.descriptor_names}"

    def calculate_descriptors(self, smiles : str) -> pd.Series:
        """Dynamically calculates all descriptors for a given compound."""
        mol = Chem.MolFromSmiles(smiles)
        if not mol:
            raise ValueError(f"Invalid SMILES: {smiles}")

        # Calculate 2D descriptors
        # 3D descriptors are not calculated, since graphs (should) capture their properties quite well already. If necessary, class can be extended.
        
        descriptors = pd.Series()

        for name in self.descriptor_names:
            if hasattr(Descriptors, name):
                descriptors[name] = getattr(Descriptors, name)(mol)
            else:
                raise ValueError(f"Descriptor {name} not found")
        
        return descriptors

if __name__ == "__main__":
    # Test the Compound class
    smis = ["CCO", "CCN", "CCO", "CCN"]
    c = [Compound(smi) for smi in smis]

    # Check __str__
    print([str(mol) for mol in c])

    # Test the DescriptorCalculator class
    descriptors = ["MolWt", "TPSA"]
    calc = DescriptorCalculator(descriptors)
    print(str(calc))
    for mol in c:
        print(calc.calculate_descriptors(mol.smiles))