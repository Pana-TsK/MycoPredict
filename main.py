import pandas as pd
import numpy as np
import logging

from src.preprocess import Preprocess
from src.dataloader import DataLoader
from src.exporter import ResultsExporter
from src.predict import Predict

from chemprop.featurizers.molecule import MorganBinaryFeaturizer

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Main:
    def __init__(self, molecular_featurizer: object = MorganBinaryFeaturizer()):
        """
        Initialize the main class for the pipeline.
        """
        self.dataloader = None
        self.smiles = None

        self.preprocessor = None
        self.datapoints = None

        self.predictor = None
        self.results = None

        self.exporter = None
        self.dataframe = None

        self.molecular_featurizer = molecular_featurizer

    def __call__(self, input_path: str, output_path: str = None, smi_col_name: str = 'smiles') -> pd.DataFrame:
        """
        Main pipeline execution.
        """
        try:
            self.smiles = self.load_data(input_path, smi_col_name)
            logger.info(f'Loaded SMILES data: {self.smiles[:5]}')  # Print first few smiles for debugging

            self.datapoints = self.preprocess_data(self.smiles)
            logger.info(f'Preprocessed {len(self.datapoints)} datapoints.')

            self.results = self.predict(self.datapoints)
            logger.info('Prediction complete.')

            self.dataframe = self.export_to_csv(output_path)
            logger.info(f'Results exported to: {output_path}')

            return self.dataframe

        except Exception as e:
            logger.error(f"Error in pipeline execution: {e}")
            raise

    def load_data(self, data_path: str, smi_col_name: str = 'smiles') -> list:
        """
        Load data from the specified path.
        """
        try:
            self.dataloader = DataLoader(data_path, smi_col_name)
            smiles = self.dataloader()
            logger.info(f"Data loaded with {len(smiles)} SMILES entries.")
            return smiles
        except Exception as e:
            logger.error(f"Error loading data from {data_path}: {e}")
            raise

    def preprocess_data(self, smiles: list) -> list:
        """
        Preprocess the SMILES data into feature vectors.
        """
        try:
            self.preprocessor = Preprocess(smiles, self.molecular_featurizer)
            datapoints = self.preprocessor()
            logger.info(f"Preprocessing completed with {len(datapoints)} datapoints.")
            return datapoints
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            raise

    def predict(self, datapoints: list) -> list:
        """
        Make predictions based on the datapoints.
        """
        try:
            self.predictor = Predict()
            results = self.predictor(datapoints)
            logger.info("Prediction finished.")
            return results
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            raise

    def export_to_csv(self, output_path: str = None) -> pd.DataFrame:
        """
        Export the results to a CSV file.
        If no output path is provided, use the default path.
        """
        try:
            if not output_path:
                output_path = 'results.csv'  # Default file name if none provided

            self.exporter = ResultsExporter(self.smiles, self.results, output_path)
            self.dataframe = self.exporter()
            logger.info(f"Results exported to: {output_path}")
            return self.dataframe
        except Exception as e:
            logger.error(f"Error exporting results to CSV: {e}")
            raise

if __name__ == '__main__':
    main_instance = Main()

    data_path = 'test_data/test_file.csv'
    output_path = 'test_data/results.csv'  # You can leave this empty if you prefer default

    try:
        whole_pipeline = main_instance(input_path=data_path, output_path=output_path)
        logger.info(f"Pipeline executed successfully. Output stored at {output_path}")
        print(whole_pipeline)
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
