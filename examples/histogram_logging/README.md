# MLflow Histogram Logging Example

This example demonstrates how to use MLflow's histogram logging feature to track distributions of model weights, gradients, and activations during training.

## Overview

MLflow's histogram logging allows you to:
- Track how weight distributions evolve during training
- Monitor gradient distributions to detect vanishing/exploding gradients
- Visualize activation distributions across layers
- Log pre-computed histograms from other frameworks (e.g., TensorBoard)

## Features

- **Automatic histogram computation**: Pass raw values and MLflow computes the histogram
- **Pre-computed histograms**: Log histograms you've already computed
- **Time-stepped logging**: Track histogram evolution across training steps
- **Hierarchical organization**: Use slashes in histogram names (e.g., `weights/layer1/bias`)

## Installation

Install MLflow with PyTorch:

```bash
pip install mlflow torch
```

## Running the Example

```bash
python train_with_histograms.py
```

This will:
1. Train a simple neural network
2. Log weight and gradient histograms at each epoch
3. Save histograms as artifacts in your MLflow run

## Viewing Histograms

After running the example:

```bash
mlflow ui
```

Then open http://localhost:5000 in your browser and navigate to the run. Histograms are stored as JSON files in the `artifacts/histograms/` directory.

## API Usage

### Log histogram from raw values

```python
import mlflow
import numpy as np

with mlflow.start_run():
    # Log weight distribution
    weights = model.get_weights().flatten()
    mlflow.log_histogram(weights, key="weights/layer1", step=0)
```

### Log pre-computed histogram

```python
import mlflow

with mlflow.start_run():
    # Pre-computed histogram bins and counts
    bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    counts = [10, 25, 30, 20, 15]

    mlflow.log_histogram(
        bins=bin_edges,
        counts=counts,
        key="activations/relu1",
        step=100
    )
```

### Log multiple histograms over training

```python
import mlflow
import torch

model = torch.nn.Linear(10, 5)

with mlflow.start_run():
    for epoch in range(10):
        # ... training code ...

        # Log weight distributions at each epoch
        for name, param in model.named_parameters():
            weights = param.detach().cpu().numpy().flatten()
            mlflow.log_histogram(weights, key=f"weights/{name}", step=epoch)
```

## Histogram Storage Format

Histograms are stored as JSON files in `artifacts/histograms/{name}.json`:

```json
[
  {
    "name": "weights/layer1",
    "step": 0,
    "timestamp": 1234567890,
    "bin_edges": [0.0, 0.1, 0.2, ...],
    "counts": [10, 25, 30, ...],
    "min_value": -0.5,
    "max_value": 0.5
  },
  {
    "name": "weights/layer1",
    "step": 1,
    "timestamp": 1234567900,
    ...
  }
]
```

## Use Cases

### 1. Detecting Vanishing/Exploding Gradients

```python
def log_gradient_histograms(model, step):
    for name, param in model.named_parameters():
        if param.grad is not None:
            grads = param.grad.detach().cpu().numpy().flatten()
            mlflow.log_histogram(grads, key=f"gradients/{name}", step=step)
```

### 2. Monitoring Weight Initialization

```python
with mlflow.start_run():
    model = create_model()

    # Log initial weight distributions
    for name, param in model.named_parameters():
        if "weight" in name:
            weights = param.detach().cpu().numpy().flatten()
            mlflow.log_histogram(weights, key=f"init_weights/{name}", step=0)
```

### 3. Comparing Different Layers

```python
with mlflow.start_run():
    # Log histograms for each layer
    for layer_idx, layer in enumerate(model.layers):
        weights = layer.weight.detach().cpu().numpy().flatten()
        mlflow.log_histogram(
            weights,
            key=f"weights/layer_{layer_idx}",
            step=epoch
        )
```

## Migrating from TensorBoard

If you're migrating from TensorBoard, you can convert TensorBoard histogram calls:

```python
# TensorBoard
writer.add_histogram('weights/layer1', weights, global_step=step)

# MLflow equivalent
mlflow.log_histogram(weights, key='weights/layer1', step=step)
```

For pre-computed TensorBoard histograms:

```python
# If you have TensorBoard histogram data
bins, counts = tensorboard_histogram_to_bins_counts(tb_histogram)

# Log in MLflow
mlflow.log_histogram(bins=bins, counts=counts, key='weights/layer1', step=step)
```

## Notes

- Histograms are stored as artifacts, not in the relational database
- Each histogram file can contain data from multiple steps
- Histogram names with slashes (e.g., `weights/layer1/bias`) are sanitized for file storage
- Default number of bins is 30, customizable via `num_bins` parameter
