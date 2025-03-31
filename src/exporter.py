import pandas as pd
import numpy as np

class ResultsExporter:
    def __init__(self, smiles : list, results : np.array, output_path : str = None):
        """
        Initialize the ResultsExporter class.
        """
        self.smiles = smiles
        self.results = results
        self.output_path = output_path if output_path else None
        
        self.df = None
    
    def __str__(self):
        return self.df

    def __call__(self):
        """
        Call the class to create a dataframe and export it to CSV.
        """
        self.df = self.create_dataframe(self.smiles, self.results, reverse=True)
        if self.output_path:
            
            self.export_to_csv(self.df, self.output_path)
        else:
            print("Output path not provided. Dataframe will not be exported.")

        return self.df

    def create_dataframe(self, smiles, results, reverse=False):
        """
        Create a sorted pandas dataframe from smiles and results. Sort by results.
        """
        df = pd.DataFrame({"smiles": smiles, "results": results})
        df.sort_values(by=["results"], ascending=reverse, inplace=True)
        df.reset_index(drop=True, inplace=True)
        return df

    def export_to_csv(self, df, output_path):
        """
        Export the dataframe to a CSV file.
        """
        df.to_csv(output_path, index=False)
        


if __name__ == "__main__":
    # Example usage
    smiles = ["CCO", "CCN", "CCOCC", "C1=CC=CC=C1"]
    results = np.array([0.77, 0.15, 0.99, 0.12])  # Random results for demonstration

    exporter = ResultsExporter(smiles, results)()
    print(exporter)