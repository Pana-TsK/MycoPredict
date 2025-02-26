import pandas as pd
import numpy as np
from lightning import pytorch as pl
from pathlib import Path

from chemprop import data, featurizers, models
from descriptor_calc import Dataset

model_path = r"C:\Users\panag\OneDrive\Documents\coding\Projects\MycoPredict\data\mlruns_2\438633058360495807\e56d439d060e4ddd90dffb56fff12273\checkpoints\best_model.ckpt"
mpnn = models.MPNN.load_from_checkpoint(model_path)

dataset_path = r"C:\Users\panag\OneDrive\Documents\coding\Projects\MycoPredict\data\validation_data\val_dataset.csv"
df = pd.read_csv(dataset_path)
print(df.shape)


