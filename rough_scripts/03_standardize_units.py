import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

# Load the dataset
df = pd.read_csv(r'C:\Users\panag\OneDrive\Documents\coding\Projects\AIbiotics\mycobacteria_ml_project\training_data\02_data_clean\02_AC50_IC50_EC50.csv')

# Function to calculate molecular weight from SMILES
def calculate_mol_weight(smiles):
    mol = Chem.MolFromSmiles(smiles)  # Convert SMILES to RDKit molecule object
    if mol is not None:
        return Descriptors.MolWt(mol)  # Calculate molecular weight
    else:
        return None  # Return None for invalid SMILES

# Add a column for molecular weights
df['molecular_weight'] = df['canonical_smiles'].apply(calculate_mol_weight)

# Define a dictionary for unit conversion to µM
unit_conversion = {
    'uM': 1,         # 1 µM = 1 µM (no conversion needed)
    'nM': 0.001,     # 1 nM = 0.001 µM
    'um': 1,         # Assuming 'um' is a typo and should be 'uM' (treated as µM)
    'mM': 1000       # 1 mM = 1000 µM
}

# Function to convert values to µM
def convert_to_uM(row):
    value = row['value']
    unit = row['units']
    if unit in unit_conversion:
        return value * unit_conversion[unit]
    elif unit == 'ug ml-1':
        # Convert ug/ml to µM using molecular weight
        molecular_weight = row['molecular_weight']
        if pd.notna(molecular_weight) and molecular_weight > 0:
            return (value / molecular_weight) * 1000
        else:
            return None  # Skip if molecular weight is missing or invalid
    else:
        return None  # Handle unknown units (if any)

# Apply the conversion to the dataframe
df['standardized_value'] = df.apply(convert_to_uM, axis=1)

# Check the results
print(df[['value', 'units', 'molecular_weight', 'standardized_value']].head())

# Save the standardized dataframe to a csv file

df.to_csv(r'training_data\02_data_clean\03_standardized_data.csv')