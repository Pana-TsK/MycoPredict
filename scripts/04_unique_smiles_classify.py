import pandas as pd

data_path = r'C:\Users\panag\OneDrive\Documents\coding\Projects\AIbiotics\mycobacteria_ml_project\training_data\02_data_clean\03_standardized_data.csv'
df = pd.read_csv(data_path)
# Assuming 'canonical_smiles' is the unique identifier and 'value' contains IC50/AC50/EC50 values
# Ensure the values are numeric
df["standardized_value"] = pd.to_numeric(df["standardized_value"], errors="coerce")

# Take the most potent value (minimum value per compound)
df_best = df.groupby("canonical_smiles")["standardized_value"].min().reset_index()

# Define hit/miss classification based on threshold (adjust as needed)
df_best["class"] = df_best["standardized_value"].apply(lambda x: 1 if x < 8 else 0)

# Filter out unwanted columns, if needed, using a mask in the next step:
df_final = df_best

# Check class balance
print(df_final["class"].value_counts())

# Save cleaned dataset
df_final.to_csv(r"C:\Users\panag\OneDrive\Documents\coding\Projects\AIbiotics\mycobacteria_ml_project\training_data\02_data_clean\04_hitmiss_reduced.csv", index=False)
