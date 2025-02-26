import duckdb
import pandas as pd

# Connect to DuckDB and query data
def query_chembl_data(db_file, bacterial_species, limit=20000, stop_after_rows=None, min_confidence=6):
    # Connect to the ChEMBL database
    con = duckdb.connect(db_file)
    con.execute("INSTALL sqlite;")
    con.execute("LOAD sqlite;")

    # Get the total number of rows with filtering criteria
    query_count = """
    SELECT COUNT(*) 
    FROM COMPOUND_STRUCTURES cs
    LEFT JOIN ACTIVITIES a ON cs.MOLREGNO = a.MOLREGNO
    LEFT JOIN ASSAYS ass ON a.ASSAY_ID = ass.ASSAY_ID
    LEFT JOIN TARGET_DICTIONARY td ON ass.TID = td.TID
    LEFT JOIN CONFIDENCE_SCORE_LOOKUP csl ON ass.CONFIDENCE_SCORE = csl.CONFIDENCE_SCORE
    WHERE td.ORGANISM = ?
    AND csl.CONFIDENCE_SCORE >= ?
    """
    total_rows = con.execute(query_count, (bacterial_species, min_confidence)).fetchone()[0]
    print(f"Total rows matching '{bacterial_species}' with confidence score ≥ {min_confidence}: {total_rows}")

    processed_rows = 0
    while processed_rows < total_rows:
        # Query data in chunks
        query = f"""
SELECT 
    cs.MOLREGNO, 
    cs.CANONICAL_SMILES, 
    cs.STANDARD_INCHI, 
    cs.STANDARD_INCHI_KEY,
    md.MOLECULE_TYPE,  
    a.TYPE, 
    a.VALUE, 
    a.UNITS,
    a.TYPE as ACTIVITY_TYPE,
    ass.ASSAY_ID,
    ass.DESCRIPTION,
    td.ORGANISM,
    csl.CONFIDENCE_SCORE  -- Added confidence score
    
FROM 
    COMPOUND_STRUCTURES cs
LEFT JOIN 
    ACTIVITIES a ON cs.MOLREGNO = a.MOLREGNO
LEFT JOIN 
    ASSAYS ass ON a.ASSAY_ID = ass.ASSAY_ID
LEFT JOIN 
    TARGET_DICTIONARY td ON ass.TID = td.TID
LEFT JOIN 
    MOLECULE_DICTIONARY md ON cs.MOLREGNO = md.MOLREGNO  
LEFT JOIN 
    CONFIDENCE_SCORE_LOOKUP csl ON ass.CONFIDENCE_SCORE = csl.CONFIDENCE_SCORE  -- Added join
WHERE 
    td.ORGANISM = ?  
    AND csl.CONFIDENCE_SCORE >= ?
ORDER BY 
    cs.MOLREGNO
LIMIT {limit}
OFFSET {processed_rows}
"""
        df_chunk = con.execute(query, (bacterial_species, min_confidence)).fetchdf()
        yield df_chunk

        # Update the count of processed rows
        processed_rows += len(df_chunk)
        print(f"Processed {processed_rows}/{total_rows} rows.")

        # Check if we need to stop early
        if stop_after_rows is not None and processed_rows >= stop_after_rows:
            print(f"Stopping early after processing {processed_rows} rows.")
            break


# Main processing function
def main():
    print(f"Querying the database:")
    db_file = "C:/Users/panag/OneDrive/Documents/coding/SharedLib/chembl_35_sqlite/chembl_35.db"
    bacterial_species = 'Mycobacterium tuberculosis'
    min_confidence = 6 # Adjust confidence score threshold

    # Combine all data chunks into a single DataFrame
    data_frames = []

    for df_chunk in query_chembl_data(db_file, bacterial_species, min_confidence=min_confidence):
        data_frames.append(df_chunk)

    # Concatenate all chunks
    full_data = pd.concat(data_frames, ignore_index=True)
    print(full_data.head())
    print(f"Processed {len(full_data)} rows in total.")
    return full_data


if __name__ == "__main__":
    data = main()
    data.to_csv('data_clean/curated_assays_MTb.csv', index=False)
