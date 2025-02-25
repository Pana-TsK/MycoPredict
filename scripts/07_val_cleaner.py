# Data pulled from 01_fetch_chembl.py will be reused
# A number of compounds have not been included, as only one assay type been used to create the model
# Therefore, an other assay type can be used to validate the model
# While the 'cut-off' value is rather vague, we can optimize for best hit-ratio
# This is likely not the ideal way to validate this model, but it should give us an idea

import pandas as pd
import chemprop

# Start with fetching the used data
# For this step, we can reuse the fetch_chembl code

df = pd.read_csv(r'data\training_data\01_curated_assays_MTb.csv')

# The ´´´activity´´´ column was chosen to validate the model
val_data = df['ACTIVITY_TYPE'] == 'Activity'
df = df[val_data]

# Reduce to only the inputs which contain percentages, and a non-null value
df = df[df['units'] == '%']
df = df[df['value'].notnull()]

# turn this val_data into a csv file with only the necessary columns
df[['canonical_smiles', 'value', 'units']].to_csv(r'C:\Users\panag\OneDrive\Documents\coding\Projects\MycoPredict\data\validation_data\val_dataset')