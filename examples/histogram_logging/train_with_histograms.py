"""
Example demonstrating MLflow histogram logging for tracking weight and gradient distributions.

This example shows how to:
1. Log histogram distributions from model weights during training
2. Log histogram distributions from gradients
3. Log histogram distributions from activations
4. Use both raw values and pre-computed histograms

Run this example:
    python train_with_histograms.py
"""

import mlflow
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Simple neural network for demonstration
class SimpleNet(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, output_size=2):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def create_dummy_dataset(n_samples=1000, input_size=10):
    """Create a simple dummy dataset for demonstration."""
    X = torch.randn(n_samples, input_size)
    y = torch.randint(0, 2, (n_samples,))
    return TensorDataset(X, y)


def log_weight_histograms(model, step):
    """Log histograms of all model weights."""
    for name, param in model.named_parameters():
        if "weight" in name:
            # Get weight values as numpy array
            weights = param.detach().cpu().numpy().flatten()
            # Log histogram with hierarchical naming
            mlflow.log_histogram(weights, key=f"weights/{name}", step=step)


def log_gradient_histograms(model, step):
    """Log histograms of all gradients."""
    for name, param in model.named_parameters():
        if param.grad is not None:
            # Get gradient values as numpy array
            grads = param.grad.detach().cpu().numpy().flatten()
            # Log histogram
            mlflow.log_histogram(grads, key=f"gradients/{name}", step=step)


def train_with_histogram_logging():
    """Train a simple model while logging histogram distributions."""
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Hyperparameters
    input_size = 10
    hidden_size = 20
    output_size = 2
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.001

    # Create model and optimizer
    model = SimpleNet(input_size, hidden_size, output_size)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create dataset and dataloader
    dataset = create_dummy_dataset(n_samples=1000, input_size=input_size)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Start MLflow run
    mlflow.set_experiment("histogram-logging-demo")

    with mlflow.start_run():
        # Log hyperparameters
        mlflow.log_params(
            {
                "input_size": input_size,
                "hidden_size": hidden_size,
                "output_size": output_size,
                "batch_size": batch_size,
                "learning_rate": learning_rate,
                "num_epochs": num_epochs,
            }
        )

        print("Training started...")
        print("Logging weight and gradient histograms at each epoch")
        print("-" * 60)

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            num_batches = 0

            # Training loop
            model.train()
            for batch_idx, (data, target) in enumerate(dataloader):
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

            # Calculate average loss
            avg_loss = epoch_loss / num_batches

            # Log scalar metrics
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            # Log weight histograms
            log_weight_histograms(model, step=epoch)

            # Log gradient histograms (compute a forward/backward pass for this)
            model.zero_grad()
            sample_data, sample_target = next(iter(dataloader))
            sample_output = model(sample_data)
            sample_loss = criterion(sample_output, sample_target)
            sample_loss.backward()
            log_gradient_histograms(model, step=epoch)

            print(
                f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f} - "
                f"Logged {len(list(model.named_parameters()))} weight histograms"
            )

        print("-" * 60)
        print(f"Training complete! Run ID: {mlflow.active_run().info.run_id}")
        print(f"\nView your histograms in the MLflow UI:")
        print(f"  mlflow ui")
        print(f"\nHistograms are stored in: artifacts/histograms/")


def example_precomputed_histograms():
    """Example showing how to log pre-computed histograms (e.g., from TensorBoard)."""
    mlflow.set_experiment("precomputed-histogram-demo")

    with mlflow.start_run():
        # Simulate pre-computed histogram data (e.g., from TensorBoard format)
        # This is useful if you're migrating from TensorBoard or have
        # custom histogram computation logic

        # Example 1: Histogram with uniform bins
        bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        counts = [5, 12, 23, 45, 67, 54, 32, 18, 8, 3]

        mlflow.log_histogram(
            bins=bin_edges, counts=counts, key="activations/relu1", step=0
        )

        # Example 2: Histogram with non-uniform bins (e.g., for log-scale)
        bin_edges = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        counts = [100, 500, 800, 300, 50]

        mlflow.log_histogram(
            bins=bin_edges, counts=counts, key="learning_rates", step=0
        )

        print("Logged pre-computed histograms")
        print(f"Run ID: {mlflow.active_run().info.run_id}")


if __name__ == "__main__":
    print("=" * 60)
    print("MLflow Histogram Logging Example")
    print("=" * 60)
    print()

    # Example 1: Train model with automatic histogram logging
    print("Example 1: Training with automatic histogram logging")
    print()
    train_with_histogram_logging()

    print()
    print("=" * 60)
    print()

    # Example 2: Log pre-computed histograms
    print("Example 2: Logging pre-computed histograms")
    print()
    example_precomputed_histograms()

    print()
    print("=" * 60)
    print("Examples complete!")
    print()
    print("To view your histograms:")
    print("  1. Run: mlflow ui")
    print("  2. Open http://localhost:5000 in your browser")
    print("  3. Navigate to your experiments and view the histogram artifacts")
    print("=" * 60)
