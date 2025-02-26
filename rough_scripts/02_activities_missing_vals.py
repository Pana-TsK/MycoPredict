import pandas as pd

def filter_data(df : pd.DataFrame, assay_types : list):
    df_filtered = df[df["ACTIVITY_TYPE"].isin(assay_types)]
    return df_filtered

def drop_missing_values(df : pd.DataFrame, columns : list):
    df_filtered = df.dropna(subset=columns)
    return df_filtered

def main():
    rawdata_path = r"C:\Users\panag\OneDrive\Documents\coding\Projects\AIbiotics\mycobacteria_ml_project\training_data\01_data_raw\01_curated_assays_MTb.csv"
    data = pd.read_csv(rawdata_path)

    # Filter for AC50, IC50, and EC50 assays. This results in approximately 9,000 rows of data.
    assay_types = ["AC50", "IC50", "EC50", "AC50_uM"]
    filtered_data = filter_data(data, assay_types)

    print(f"Retained {len(filtered_data)} rows with activity types {assay_types}.")

    # remove all rows with missing values for canoncial_smiles, standard_value, and standard_units
    columns = ["canonical_smiles", "value", "units"]
    filtered_data = drop_missing_values(filtered_data, columns)
    print(f"Retained {len(filtered_data)} rows after removing missing values.")

    return filtered_data

if __name__ == "__main__":
    data = main()
    assay_types = "AC50_IC50_EC50"
    print(data.head())
    print(f"Processed {len(data)} rows in total.")
    data.to_csv(r"training_data\02_data_clean\02_AC50_IC50_EC50.csv", index=False)