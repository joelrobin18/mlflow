"""Configuration management for Codex CLI integration."""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
)

# ============================================================================
# CONSTANTS
# ============================================================================

# Configuration field names
MLFLOW_CODEX_TRACING_ENABLED = "MLFLOW_CODEX_TRACING_ENABLED"
SESSIONS_DIR_FIELD = "sessions_dir"
ENVIRONMENT_FIELD = "environment"

# Default paths
DEFAULT_CODEX_DIR = Path.home() / ".codex"
DEFAULT_SESSIONS_DIR = DEFAULT_CODEX_DIR / "sessions"
DEFAULT_CONFIG_FILE = DEFAULT_CODEX_DIR / "mlflow_config.json"


# ============================================================================
# CONFIGURATION DATACLASS
# ============================================================================


@dataclass
class CodexTracingStatus:
    """Status of Codex tracing configuration."""

    enabled: bool
    tracking_uri: str | None = None
    experiment_id: str | None = None
    experiment_name: str | None = None
    sessions_dir: str | None = None
    reason: str | None = None


# ============================================================================
# CONFIGURATION MANAGEMENT
# ============================================================================


def get_config_path() -> Path:
    """Get the path to MLflow Codex configuration file."""
    return DEFAULT_CONFIG_FILE


def get_sessions_dir() -> Path:
    """Get the path to Codex sessions directory."""
    config = load_config()
    if config and SESSIONS_DIR_FIELD in config:
        return Path(config[SESSIONS_DIR_FIELD])
    return DEFAULT_SESSIONS_DIR


def load_config() -> dict[str, Any]:
    """Load MLflow Codex configuration from file."""
    config_path = get_config_path()
    if not config_path.exists():
        return {}

    try:
        with open(config_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return {}


def save_config(config: dict[str, Any]) -> None:
    """Save MLflow Codex configuration to file."""
    config_path = get_config_path()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def setup_environment_config(
    tracking_uri: str | None = None,
    experiment_id: str | None = None,
    experiment_name: str | None = None,
    sessions_dir: str | None = None,
) -> None:
    """Set up environment configuration for Codex tracing.

    Args:
        tracking_uri: MLflow tracking URI
        experiment_id: MLflow experiment ID
        experiment_name: MLflow experiment name
        sessions_dir: Custom Codex sessions directory path
    """
    config = load_config()

    if ENVIRONMENT_FIELD not in config:
        config[ENVIRONMENT_FIELD] = {}

    # Enable tracing
    config[ENVIRONMENT_FIELD][MLFLOW_CODEX_TRACING_ENABLED] = "true"

    # Set tracking URI (default to local file storage)
    if tracking_uri:
        config[ENVIRONMENT_FIELD][MLFLOW_TRACKING_URI.name] = tracking_uri
    elif MLFLOW_TRACKING_URI.name not in config[ENVIRONMENT_FIELD]:
        # Default to standard MLflow directory (./mlruns in current directory)
        # This makes traces visible with just 'mlflow ui' without extra flags
        import os
        default_uri = f"file://{os.path.abspath('./mlruns')}"
        config[ENVIRONMENT_FIELD][MLFLOW_TRACKING_URI.name] = default_uri

    # Set experiment configuration
    if experiment_id:
        config[ENVIRONMENT_FIELD][MLFLOW_EXPERIMENT_ID.name] = experiment_id
        # Remove experiment_name if both are provided
        config[ENVIRONMENT_FIELD].pop(MLFLOW_EXPERIMENT_NAME.name, None)
    elif experiment_name:
        config[ENVIRONMENT_FIELD][MLFLOW_EXPERIMENT_NAME.name] = experiment_name
        # Remove experiment_id if both are provided
        config[ENVIRONMENT_FIELD].pop(MLFLOW_EXPERIMENT_ID.name, None)

    # Set custom sessions directory if provided
    if sessions_dir:
        config[SESSIONS_DIR_FIELD] = str(Path(sessions_dir).resolve())

    save_config(config)


def get_env_var(name: str, default: str = "") -> str:
    """Get environment variable from config or system environment.

    Args:
        name: Environment variable name
        default: Default value if not found

    Returns:
        Environment variable value
    """
    # First check system environment
    if name in os.environ:
        return os.environ[name]

    # Then check config file
    config = load_config()
    if ENVIRONMENT_FIELD in config and name in config[ENVIRONMENT_FIELD]:
        return config[ENVIRONMENT_FIELD][name]

    return default


def is_tracing_enabled() -> bool:
    """Check if Codex tracing is enabled."""
    value = get_env_var(MLFLOW_CODEX_TRACING_ENABLED, "false")
    return value.lower() in ("true", "1", "yes")


def get_tracing_status() -> CodexTracingStatus:
    """Get current Codex tracing status.

    Returns:
        CodexTracingStatus with current configuration
    """
    config = load_config()

    if not config or ENVIRONMENT_FIELD not in config:
        return CodexTracingStatus(
            enabled=False, reason="No configuration found - run 'mlflow autolog codex' to enable"
        )

    env_config = config[ENVIRONMENT_FIELD]

    if not env_config.get(MLFLOW_CODEX_TRACING_ENABLED, "false").lower() in (
        "true",
        "1",
        "yes",
    ):
        return CodexTracingStatus(enabled=False, reason="Tracing is disabled in configuration")

    tracking_uri = env_config.get(MLFLOW_TRACKING_URI.name)
    experiment_id = env_config.get(MLFLOW_EXPERIMENT_ID.name)
    experiment_name = env_config.get(MLFLOW_EXPERIMENT_NAME.name)
    sessions_dir = config.get(SESSIONS_DIR_FIELD, str(DEFAULT_SESSIONS_DIR))

    return CodexTracingStatus(
        enabled=True,
        tracking_uri=tracking_uri,
        experiment_id=experiment_id,
        experiment_name=experiment_name,
        sessions_dir=sessions_dir,
    )


def disable_tracing() -> bool:
    """Disable Codex tracing.

    Returns:
        True if configuration was removed, False if no configuration found
    """
    config_path = get_config_path()

    if not config_path.exists():
        return False

    config_path.unlink()
    return True
