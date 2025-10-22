"""
Quick Start Example: Using MLflow with uv Package Manager

This example demonstrates how to use MLflow's automatic requirements inference
with uv package manager for training and logging ML models.

Prerequisites:
1. Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh
2. Set up project:
   $ uv init my-ml-project
   $ cd my-ml-project
   $ uv add mlflow scikit-learn pandas numpy
3. Run this script:
   $ uv run python uv_quickstart.py

MLflow will automatically detect your uv.lock file and infer requirements!
"""

import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris, load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import numpy as np


def train_and_log_model():
    """
    Train a simple Random Forest model and log it with MLflow.
    MLflow will automatically infer requirements from uv.lock!
    """
    print("=" * 80)
    print("MLflow + uv Package Manager - Quick Start Example")
    print("=" * 80)

    # Set up MLflow
    mlflow.set_experiment("uv-quickstart-experiment")

    # Load data
    print("\n1. Loading Iris dataset...")
    X, y = load_iris(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Start MLflow run
    with mlflow.start_run(run_name="rf-iris-uv-example") as run:
        print(f"\n2. Starting MLflow run: {run.info.run_id}")

        # Train model
        print("\n3. Training Random Forest model...")
        model = RandomForestClassifier(
            n_estimators=100, max_depth=5, random_state=42
        )
        model.fit(X_train, y_train)

        # Evaluate
        print("\n4. Evaluating model...")
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")

        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   F1 Score: {f1:.4f}")

        # Log parameters
        mlflow.log_param("n_estimators", 100)
        mlflow.log_param("max_depth", 5)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_score", f1)

        # Log model with automatic uv requirements inference!
        print("\n5. Logging model with automatic uv requirements inference...")
        print("   MLflow will:")
        print("   - Detect uv binary")
        print("   - Find uv.lock file")
        print("   - Run: uv export --no-dev --no-hashes --no-editable --frozen --locked")
        print("   - Extract and save requirements.txt")

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            input_example=X_train[:5],
            # NOTE: No pip_requirements specified!
            # MLflow automatically infers from uv.lock
        )

        print("\n6. Model logged successfully!")
        print(f"   Run ID: {run.info.run_id}")
        print(f"   Artifact URI: {run.info.artifact_uri}")

        # Get and display inferred requirements
        model_uri = f"runs:/{run.info.run_id}/model"

        return run.info.run_id, model_uri


def verify_requirements(model_uri):
    """
    Verify what requirements were inferred and logged.
    """
    print("\n" + "=" * 80)
    print("Verifying Inferred Requirements")
    print("=" * 80)

    from mlflow.pyfunc import get_model_dependencies

    requirements = get_model_dependencies(model_uri)

    print("\n✅ Requirements successfully inferred from uv.lock:")
    print("-" * 80)
    for req in requirements:
        print(f"   {req}")
    print("-" * 80)

    return requirements


def load_and_test_model(model_uri):
    """
    Load the logged model and test inference.
    """
    print("\n" + "=" * 80)
    print("Loading and Testing Model")
    print("=" * 80)

    # Load model
    print(f"\n1. Loading model from: {model_uri}")
    model = mlflow.pyfunc.load_model(model_uri)

    # Test prediction
    print("\n2. Testing inference...")
    X_test, _ = load_iris(return_X_y=True)
    X_test = X_test[:5]  # Just test on 5 samples

    predictions = model.predict(X_test)

    print("\n   Sample predictions:")
    print(f"   Input shape: {X_test.shape}")
    print(f"   Predictions: {predictions}")
    print("\n✅ Model loaded and inference successful!")


def demonstrate_serving_command(run_id):
    """
    Show how to serve the model.
    """
    print("\n" + "=" * 80)
    print("Model Serving Examples")
    print("=" * 80)

    print("\n1. Serve model locally:")
    print(f"   $ mlflow models serve -m 'runs:/{run_id}/model' -p 5000")
    print("\n   Then test with curl:")
    print("   $ curl http://127.0.0.1:5000/invocations -H 'Content-Type: application/json' \\")
    print('        -d \'{"dataframe_split": {"columns": ["c0","c1","c2","c3"], "data": [[5.1,3.5,1.4,0.2]]}}\'')

    print("\n2. Build Docker image:")
    print(f"   $ mlflow models build-docker -m 'runs:/{run_id}/model' -n 'iris-model'")
    print("   $ docker run -p 5001:8080 iris-model")

    print("\n3. Deploy to cloud platforms:")
    print("   - Databricks: Use Model Serving with the registered model")
    print("   - AWS SageMaker: Use mlflow.sagemaker.deploy()")
    print("   - Azure ML: Use mlflow deployments")

    print("\n💡 All deployments will use the uv-inferred requirements.txt!")


def demonstrate_model_registry(run_id):
    """
    Show how to register the model.
    """
    print("\n" + "=" * 80)
    print("Model Registry Example")
    print("=" * 80)

    print("\n1. Register the model:")
    print(f"   $ mlflow models register -m 'runs:/{run_id}/model' -n 'iris-classifier'")

    print("\n2. Or programmatically:")
    print(f"""
    import mlflow

    # Register model
    result = mlflow.register_model(
        model_uri="runs:/{run_id}/model",
        name="iris-classifier"
    )

    # Transition to production
    client = mlflow.MlflowClient()
    client.transition_model_version_stage(
        name="iris-classifier",
        version=result.version,
        stage="Production"
    )
    """)

    print("\n✅ The registered model includes all uv-inferred requirements!")


def main():
    """
    Main function to run the complete example.
    """
    try:
        # Check if uv is available
        import shutil

        if shutil.which("uv") is None:
            print("\n⚠️  WARNING: uv binary not found!")
            print("   Install uv: curl -LsSf https://astral.sh/uv/install.sh | sh")
            print("   This example will fall back to traditional requirements inference.\n")

        # Check if uv.lock exists
        from pathlib import Path

        uv_lock_path = None
        for directory in [Path.cwd()] + list(Path.cwd().parents):
            if (directory / "uv.lock").exists():
                uv_lock_path = directory / "uv.lock"
                break

        if uv_lock_path:
            print(f"\n✅ Found uv.lock at: {uv_lock_path}")
            print("   MLflow will use this for requirements inference!\n")
        else:
            print("\n⚠️  WARNING: No uv.lock file found!")
            print("   Run 'uv lock' to generate it.")
            print("   This example will fall back to traditional requirements inference.\n")

        # Train and log model
        run_id, model_uri = train_and_log_model()

        # Verify requirements
        requirements = verify_requirements(model_uri)

        # Load and test model
        load_and_test_model(model_uri)

        # Show serving examples
        demonstrate_serving_command(run_id)

        # Show registry examples
        demonstrate_model_registry(run_id)

        print("\n" + "=" * 80)
        print("🎉 Example completed successfully!")
        print("=" * 80)
        print("\nNext steps:")
        print("1. View the run in MLflow UI: mlflow ui")
        print(f"2. Load the model: mlflow.pyfunc.load_model('runs:/{run_id}/model')")
        print(f"3. Serve the model: mlflow models serve -m 'runs:/{run_id}/model'")
        print("4. Deploy to production!")
        print("\n💡 All deployments will automatically use your uv-managed dependencies!")
        print("=" * 80 + "\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback

        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
