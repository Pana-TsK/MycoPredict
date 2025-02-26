import pandas as pd
from pathlib import Path
from lightning import pytorch as pl
import chemprop
from chemprop import data, featurizers, models, nn
from chemprop.nn.agg import NormAggregation
import torch
import mlflow
import mlflow.pytorch

# Specify input for the model
input_path = r"C:\Users\panag\OneDrive\Documents\coding\Projects\MycoPredict\data\training_data\descriptors\05_descriptors_V2.csv"
num_workers = 0  # Number of workers for dataloader (0 uses the main process). I keep getting recommended to increase the number of workers, but this causes issues.
smiles_column = 'SMILES'  # Name of the column containing SMILES strings
target_columns = ['Hit_Miss']  # Binary classification labels (0 or 1)

# Load the input dataframe
df_input = pd.read_csv(input_path)

# Extract SMILES and targets
smis = df_input[smiles_column].values
targets = df_input[target_columns].values

# Convert to MoleculeDatapoint format
all_data = [data.MoleculeDatapoint.from_smi(smi, y) for smi, y in zip(smis, targets)]

# Scaffold-balanced train/val/test split
mols = [d.mol for d in all_data]
train_indices, val_indices, test_indices = data.make_split_indices(mols, "scaffold_balanced", (0.8, 0.1, 0.1))
train_data, val_data, test_data = data.split_data_by_indices(all_data, train_indices, val_indices, test_indices)

# Create MoleculeDataset without re-featurization
train_dset = data.MoleculeDataset(train_data[0])
val_dset = data.MoleculeDataset(val_data[0])
test_dset = data.MoleculeDataset(test_data[0])

# Dataloaders with batch size 64
batch_size = 64
train_loader = data.build_dataloader(train_dset, batch_size=batch_size, num_workers=num_workers)
val_loader = data.build_dataloader(val_dset, batch_size=batch_size, num_workers=num_workers, shuffle=False)
test_loader = data.build_dataloader(test_dset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

# Define model components with increased capacity
mp = nn.AtomMessagePassing()
agg = nn.NormAggregation()
ffn = nn.BinaryClassificationFFN(dropout=0.5)
batch_norm = True
metric_list = [
    nn.metrics.BinaryF1Score(),
    nn.metrics.BinaryMCCMetric(),
    nn.metrics.BinaryAccuracy()
]

# Construct the MPNN model
mpnn = models.MPNN(mp, agg, ffn, batch_norm, metric_list)

# Use mixed precision for faster training
torch.set_float32_matmul_precision('high')

# Define optimizer with weight decay
optimizer = torch.optim.AdamW(mpnn.parameters(), lr=1e-4, weight_decay=1e-4)

# Learning rate scheduler
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=5, factor=0.5)

# Add ModelCheckpoint to callbacks
checkpoint_cb = pl.callbacks.ModelCheckpoint(
    monitor="val_loss",
    mode="min",
    save_top_k=1,
    filename="best_model"
)

# Define PyTorch Lightning trainer
trainer = pl.Trainer(
    logger=pl.loggers.MLFlowLogger(experiment_name="training_version_2"),
    callbacks=[checkpoint_cb],
    enable_checkpointing=True,
    enable_progress_bar=True,
    accelerator="gpu",
    devices=1,
    max_epochs=100,
    min_epochs=10,
    log_every_n_steps=1,
    precision="16-mixed",
    deterministic=True
)

# Train the model
trainer.fit(mpnn, train_loader, val_loader)

# After training, load best model
best_model = models.MPNN.load_from_checkpoint(checkpoint_cb.best_model_path)

# Test the best model
results = trainer.test(best_model, test_loader)

# Log results to MLflow
for metric, value in results[0].items():
    mlflow.log_metric(metric, value)

# Save the best model to MLflow
mlflow.pytorch.log_model(best_model, r"mlruns_2\best_model")