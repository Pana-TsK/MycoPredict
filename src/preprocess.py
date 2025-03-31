from chemprop import data
from chemprop.data import MoleculeDatapoint

from chemprop.featurizers.molecule import (
    MorganBinaryFeaturizer # If you want to use a different featurizer, import it here or in the main class
)

from chemprop.data import MoleculeDatapoint
from chemprop.utils import make_mol

class Preprocess:
    """
    Preprocess a list of smiles strings into their corresponding graph and datapoint representations.
    This class is designed to work with the Chemprop library for molecular graph representation and property prediction.

    Example usage: 
    >>> preprocessor = Preprocess()
    >>> preprocessor(['CCO'])
    MoleculeDatapoint(mol=<rdkit.Chem.rdchem.Mol object at 0x000002C94ACCB530>, y=array([0, 0, 0, ..., 0, 0, 0], dtype=uint8), weight=1.0, gt_mask=None, lt_mask=None, x_d=None, x_phase=None, name='CCO', V_f=None, E_f=None, V_d=None)
    """
    def __init__(self, smiles: list, molecular_featurizer = MorganBinaryFeaturizer()):
        """
        Initialize the Preprocess class.
        >>> Preprocess(path='path/to/data.csv')
        DataFrame: pd.DataFrame
        """
        # Create the mols
        self.smiles = smiles
        self.molecular_featurizer = molecular_featurizer

        self.mols = []
        self.xds = []

        self.datapoints = []

    def __str__(self):
        return f"preprocessed smiles strings with {len(self.smiles)}" # fill in after the class is finished
    
    def __repr__(self):
        return f'Preprocess({self.smiles})'

    # create iteration for ease of use
    def __iter__(self):
        return iter(self.datapoints)

    def __next__(self):
        for datapoint in self.datapoints:
            yield datapoint
        raise StopIteration
    
    # Create call method to allow the class to be called like a function
    # This allows for easy integration with other libraries and frameworks
    # and reduces boilerplate

    def __call__(self):
        """
        Automatically run preprocessing steps when the instance is called.
        """
        self.mols = self.convert_to_mol(self.smiles)       

        self.xds = self.create_xds(self.mols)

        self.datapoints = self.create_datapoints(self.smiles, self.xds)

        return self.datapoints

    # Class methods
    
    def convert_to_mol(self, smiles):
        """
        Convert the SMILES strings into a list of RDKit molecules, filtering out any failed conversions.
        """
        mols = []
        for smi in smiles:
            mol = make_mol(smi, keep_h=False, add_h=False)
            if mol is None:
                print(f"Warning: SMILES {smi} could not be converted to a molecule.")
            else:
                mols.append(mol)
        return mols

    def create_xds(self, mols):
        """
        Create the additional datapoints for the mols.
        """
        try:
            xds = [self.molecular_featurizer(mol) for mol in mols]
        except Exception as e:
            print(f"Error creating xds: {e}")
        
        return xds

    def create_datapoints(self, smiles, xds):
        """
        Create the datapoints for the mols.
        """
        try:
            datapoints = [data.MoleculeDatapoint.from_smi(smi, x_d = xd) for smi, xd in zip(smiles, xds)]
        except Exception as e:
            print(f"Error creating datapoints: {e}")
        
        return datapoints

if __name__ == "__main__":
    # Example usage
    smiles_list = ["CCO", "CCN", "CCC"]

    preprocess = Preprocess(smiles_list)
    datapoints = preprocess()

    for datapoint in datapoints:
        print("Datapoint:")
        print(datapoint)