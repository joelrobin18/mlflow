"""Configuration management for Opencode integration with MLflow."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# MLflow environment variable constants
MLFLOW_TRACING_ENABLED = "MLFLOW_OPENCODE_TRACING_ENABLED"

# npm package name for the MLflow Opencode plugin
MLFLOW_PLUGIN_NPM_PACKAGE = "mlflow-opencode"

# Opencode config file name
OPENCODE_CONFIG_FILE = "opencode.json"


@dataclass
class TracingStatus:
    """Dataclass for tracing status information."""

    enabled: bool
    tracking_uri: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    reason: str | None = None


def get_opencode_config_path(directory: Path) -> Path:
    """Get the path to the Opencode configuration file.

    Args:
        directory: Project directory

    Returns:
        Path to opencode.json file
    """
    return directory / OPENCODE_CONFIG_FILE


def load_opencode_config(config_path: Path) -> dict[str, Any]:
    """Load existing Opencode configuration from config file.

    Args:
        config_path: Path to opencode.json file

    Returns:
        Configuration dictionary, empty dict if file doesn't exist or is invalid
    """
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def save_opencode_config(config_path: Path, config: dict[str, Any]) -> None:
    """Save Opencode configuration to config file.

    Args:
        config_path: Path to opencode.json file
        config: Configuration dictionary to save
    """
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def _is_mlflow_plugin(plugin_name: str) -> bool:
    """Check if a plugin entry is an MLflow plugin.

    Args:
        plugin_name: Plugin entry string

    Returns:
        True if this is an MLflow plugin entry
    """
    if MLFLOW_PLUGIN_NPM_PACKAGE in plugin_name:
        return True
    # Also match file:// paths from old installations
    if plugin_name.startswith("file://") and "mlflow" in plugin_name.lower():
        return True
    return False


def _get_mlflow_config_path(directory: Path) -> Path:
    """Get the path to the MLflow config file."""
    return directory / ".opencode" / "mlflow.json"


def _load_mlflow_config(directory: Path) -> dict[str, Any]:
    """Load MLflow configuration from .opencode/mlflow.json.

    Args:
        directory: Project directory

    Returns:
        Configuration dictionary, empty dict if file doesn't exist or is invalid
    """
    config_path = _get_mlflow_config_path(directory)
    if config_path.exists():
        try:
            with open(config_path, encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}
    return {}


def get_env_var(var_name: str, default: str = "") -> str:
    """Get environment variable from OS.

    Args:
        var_name: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value
    """
    value = os.getenv(var_name)
    if value is not None:
        return value
    return default


def get_tracing_status(config_path: Path) -> TracingStatus:
    """Get current tracing status from Opencode configuration.

    Args:
        config_path: Path to Opencode config file

    Returns:
        TracingStatus with tracing status information
    """
    if not config_path.exists():
        return TracingStatus(enabled=False, reason="No configuration found")

    config = load_opencode_config(config_path)

    # Check if MLflow plugin is configured
    plugins = config.get("plugin", [])
    enabled = any(_is_mlflow_plugin(p) for p in plugins)

    # Read MLflow settings from .opencode/mlflow.json (not environment variables)
    directory = config_path.parent
    mlflow_config = _load_mlflow_config(directory)

    tracking_uri = mlflow_config.get("trackingUri")
    experiment_id = mlflow_config.get("experimentId")
    experiment_name = mlflow_config.get("experimentName")

    return TracingStatus(
        enabled=enabled,
        tracking_uri=tracking_uri,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
    )


def setup_hook_config(
    config_path: Path,
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
) -> str:
    """Set up MLflow tracing plugin in Opencode configuration.

    Configures Opencode to use the mlflow-opencode npm package for tracing.
    Users need to install the package first: bun add mlflow-opencode

    Args:
        config_path: Path to Opencode config file
        tracking_uri: MLflow tracking URI
        experiment_id: MLflow experiment ID (takes precedence over name)
        experiment_name: MLflow experiment name

    Returns:
        The plugin name that was configured
    """
    config = load_opencode_config(config_path)

    # Ensure plugin array exists
    if "plugin" not in config:
        config["plugin"] = []

    # Remove any existing MLflow plugin entries
    config["plugin"] = [p for p in config["plugin"] if not _is_mlflow_plugin(p)]

    # Add the MLflow plugin npm package
    config["plugin"].append(MLFLOW_PLUGIN_NPM_PACKAGE)

    save_opencode_config(config_path, config)

    # Create MLflow config file that the TypeScript plugin will read
    _create_mlflow_config_file(config_path.parent, tracking_uri, experiment_id, experiment_name)

    return MLFLOW_PLUGIN_NPM_PACKAGE


def _create_mlflow_config_file(
    directory: Path,
    tracking_uri: str | None,
    experiment_id: str | None,
    experiment_name: str | None,
) -> None:
    """Create a JSON config file with MLflow settings for the plugin.

    Args:
        directory: Directory to create the config file in
        tracking_uri: MLflow tracking URI
        experiment_id: MLflow experiment ID
        experiment_name: MLflow experiment name
    """
    opencode_dir = directory / ".opencode"
    opencode_dir.mkdir(parents=True, exist_ok=True)

    config_file = opencode_dir / "mlflow.json"
    config_data: dict[str, Any] = {"enabled": True}

    if tracking_uri:
        config_data["trackingUri"] = tracking_uri
    if experiment_id:
        config_data["experimentId"] = experiment_id
    elif experiment_name:
        config_data["experimentName"] = experiment_name

    with open(config_file, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)


def disable_tracing(config_path: Path) -> bool:
    """Remove MLflow plugin from Opencode configuration.

    Args:
        config_path: Path to Opencode config file

    Returns:
        True if configuration was modified, False if no configuration was found
    """
    if not config_path.exists():
        return False

    config = load_opencode_config(config_path)
    modified = False

    # Remove MLflow plugin from plugin list
    if "plugin" in config:
        original_len = len(config["plugin"])
        config["plugin"] = [p for p in config["plugin"] if not _is_mlflow_plugin(p)]
        if len(config["plugin"]) != original_len:
            modified = True
        if not config["plugin"]:
            del config["plugin"]

    # Remove MLflow config file if it exists
    mlflow_config = _get_mlflow_config_path(config_path.parent)
    if mlflow_config.exists():
        mlflow_config.unlink()
        modified = True

    # Also clean up old mlflow-env.sh if it exists
    opencode_dir = config_path.parent / ".opencode"
    old_env_script = opencode_dir / "mlflow-env.sh"
    if old_env_script.exists():
        old_env_script.unlink()

    if modified:
        if config:
            save_opencode_config(config_path, config)
        else:
            config_path.unlink()

    return modified
