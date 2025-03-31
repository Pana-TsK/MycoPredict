import pandas as pd
import numpy as np

import torch
from lightning import pytorch as pl
from pathlib import Path

from chemprop import data, featurizers, models


class Predict:
    def __init__(self, model_path: str = 'src/model/mycopredict_best.ckpt', featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()):
        """
        Initialize the Predict class with the path to the model checkpoint.
        """
        self.model = models.MPNN.load_from_checkpoint(model_path)
        self.featurizer = featurizer

        self.test_dset = []
        self.test_loader = []

        self.test_preds = []
    
    def __call__(self, test_data: list):
        """
        Automatically run predictor steps when the instance is called.
        """
        self.test_dset, self.test_loader = self.build_test_dataset(test_data)
        self.initialize_model()
        self.test_preds = self.run_model()

        return self.test_preds


    def build_test_dataset(self, test_data: list, num_workers: int = 4, persistent_workers: bool = True):
        """
        Separate the datapoints created by the Preprocess class into a test dataset.
        """
        test_dset = data.MoleculeDataset(test_data, featurizer=self.featurizer)
        test_loader = data.build_dataloader(test_dset, shuffle=False, num_workers=num_workers, persistent_workers=persistent_workers)

        return test_dset, test_loader
    
    def initialize_model(self):
        """
        Control whether to use GPU or CPU for training.
        """
        with torch.inference_mode():
            self.trainer = pl.Trainer(
                logger=True,
                enable_progress_bar=True,
                accelerator="gpu",
                devices=1
            )
        try:
            torch.set_float32_matmul_precision('medium')
        except AttributeError:
            pass
    
    def run_model(self):
        test_preds = self.trainer.predict(self.model, self.test_loader)

        return test_preds[0].squeeze().tolist()

    


if __name__ == '__main__':

    # Check if model is loaded correctly
    predictor = Predict()
    print(predictor.model)

    # Create the Preprocess class and load the data
    preprocessor = Preprocess()
    smiles = ['CCO', 'CCN', 'CCC']
    datapoints = preprocessor(smiles)
    print(f"Datapoints created: {datapoints}")

    # Create the test dataset and dataloader
    predictions = predictor(datapoints)
    print(f"Predictions: {predictions}")



